import asyncio
import json
import os
import sys
import time
from distutils.util import strtobool
from typing import Any

import ckan.lib.api_token as api_token
import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from flask.views import MethodView
from loguru import logger
from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart
from pydantic_ai.usage import UsageLimits

from ckanext.chat.bot.agent import (UploadedFile, exception_to_model_response,
                                    user_input_to_model_request)
from ckanext.chat.helpers import service_available

#mp.set_start_method("spawn", force=True)
logger.remove()
if bool(strtobool(os.environ.get("DEBUG", "false"))):
    log_level = "DEBUG"
else:
    log_level = "INFO"
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | [{name}] {message}",
    level=log_level,
    enqueue=True,
)

blueprint = Blueprint("chat", __name__)

global_ckan_app = None


@blueprint.before_request
def capture_global_app():
    # This hook is executed in an active application context.
    global global_ckan_app
    if global_ckan_app is None:
        # Capture the global CKAN app from the current request's context
        global_ckan_app = current_app._get_current_object()


class ChatView(MethodView):
    def post(self):
        return core_helpers.redirect_to(
            "chat.chat",
        )

    def get(self):
        if current_user.is_anonymous:
            core_helpers.flash_error(_("Not authorized to see this page"))

            # flask types do not mention that it's possible to return a response
            # from the `before_request` callback
            return core_helpers.redirect_to("user.login")
        # logger.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )

MAX_HISTORY_MESSAGES = 100
MAX_MESSAGE_CONTENT_LENGTH = 50000
VALID_MESSAGE_KINDS = {"request", "response"}


def _authenticate_request():
    """Authenticate via API token header or CKAN session. Returns (user_id, error_response)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        token = auth_header.removeprefix("Bearer ").strip()
        if token:
            user = api_token.get_user_from_token(token)
            if user:
                return user.id, None
        return None, (jsonify({"success": False, "msg": "Invalid API token"}), 401)

    tkuser = toolkit.current_user
    if tkuser and not tkuser.is_anonymous and tkuser.id:
        return tkuser.id, None

    return None, (jsonify({"success": False, "msg": "Authentication required"}), 401)


def _extract_upload():
    """Extract uploaded file from request.files, if present."""
    f = request.files.get("upload")
    if f and f.filename:
        return UploadedFile(
            filename=f.filename,
            content_type=f.content_type or "application/octet-stream",
            data=f.read(),
        )
    return None


def _validate_history(history_str: str):
    if not history_str:
        return None
    try:
        history_list = json.loads(history_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid history JSON, ignoring")
        return None

    if not isinstance(history_list, list):
        return None
    if len(history_list) > MAX_HISTORY_MESSAGES:
        history_list = history_list[-MAX_HISTORY_MESSAGES:]

    for msg in history_list:
        if isinstance(msg, dict):
            kind = msg.get("kind", "")
            if kind not in VALID_MESSAGE_KINDS:
                logger.warning(f"Rejected history message with invalid kind: {kind}")
                return None
            for part in msg.get("parts", []):
                if isinstance(part, dict):
                    content = part.get("content", "")
                    if isinstance(content, str) and len(content) > MAX_MESSAGE_CONTENT_LENGTH:
                        part["content"] = content[:MAX_MESSAGE_CONTENT_LENGTH]

    return ModelMessagesTypeAdapter.validate_python(history_list)


def ask():
    user_id, auth_error = _authenticate_request()
    if auth_error:
        return auth_error

    user_input = request.form.get("text")
    history = request.form.get("history", "")
    _raw = request.form.get("research", "")
    research = _raw is True or (isinstance(_raw, str) and _raw.lower() in ("true", "1", "research"))
    uploaded_file = _extract_upload()
    debug = bool(strtobool(os.environ.get("DEBUG", "false")))

    try:
        response = asyncio.run(
            _agent_worker(user_input, history, user_id=user_id,
                          research=research, uploaded_file=uploaded_file),
            debug=debug,
        )
        messages = response.new_messages()
        [
            [
                message.parts.remove(part)
                for part in message.parts
                if isinstance(part, TextPart) and part.content == ""
            ]
            for message in messages
        ]
        return jsonify({"response": messages})

    except Exception as e:
        user_promt = user_input_to_model_request(user_input)
        error_response = exception_to_model_response(e)
        logger.error(error_response)
        return jsonify({"response": [user_promt, error_response]})


async def _agent_worker(prompt: str, history: str, user_id: str,
                        research: bool = False,
                        status_queue: 'asyncio.Queue | None' = None,
                        uploaded_file: 'UploadedFile | None' = None) -> Any:
    from loguru import logger as _logger
    from ckanext.chat.bot.agent import (
        Deps, agent, research_agent,
        mcp_available, get_user_token, config, _push_status,
    )
    from ckanext.chat.bot.utils import init_dynamic_models, dynamic_models_initialized

    log = _logger.bind(process="worker", user_id=user_id)
    log.debug(f"Worker starting for {user_id}")

    if not dynamic_models_initialized:
        init_dynamic_models()

    deps = Deps(user_id=user_id, status_queue=status_queue, uploaded_file=uploaded_file)
    msg_history = _validate_history(history)

    if mcp_available():
        host = toolkit.config.get("ckan.devserver.host", "localhost")
        port = toolkit.config.get("ckan.devserver.port", "5000")
        mcp_url = f"http://{host}:{port}/mcp"
        token = get_user_token(user_id)
        if token:
            deps.mcp_token = token
            deps.mcp_url = mcp_url
            log.info(f"MCP path enabled, url={mcp_url}")
        else:
            log.warning("MCP available but token creation failed, falling back to ckan_agent")

    if deps.mcp_url:
        log.info("Using MCP execution path (JSON-RPC)")
    else:
        log.info("Using ckan_agent fallback path")

    active_agent = research_agent if research else agent
    if research:
        log.info("Switching from front_agent to research_agent")
        _push_status(deps, "Switching to research agent (deep research mode)")
    limits = (
        UsageLimits(request_limit=config.REQUEST_LIMIT_RESEARCH_AGENT, total_tokens_limit=config.MAX_TOKENS_RESEARCH_AGENT)
        if research else
        UsageLimits(request_limit=config.REQUEST_LIMIT_FRONT_AGENT, total_tokens_limit=config.MAX_TOKENS_FRONT_AGENT)
    )

    r = await active_agent.run(
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
        usage_limits=limits,
    )

    log.debug(f"Worker done, result type: {type(r)}")
    await _logger.complete()
    return r


async def _stream_with_status(prompt, history, user_id, research=False,
                              uploaded_file=None):
    status_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    async def _worker():
        try:
            r = await _agent_worker(prompt, history, user_id, research,
                                    status_queue=status_queue,
                                    uploaded_file=uploaded_file)
            await result_queue.put(('result', r))
        except Exception as e:
            await result_queue.put(('error', e))
        finally:
            await status_queue.put(None)

    task = asyncio.create_task(_worker())

    _KEEPALIVE_INTERVAL = 15
    last_yield_time = time.monotonic()
    running = True
    while running:
        yielded = False
        try:
            status = await asyncio.wait_for(status_queue.get(), timeout=0.2)
            if status is None:
                running = False
            else:
                yield ('status', status)
                yielded = True
        except asyncio.TimeoutError:
            if task.done():
                running = False

        if yielded:
            last_yield_time = time.monotonic()
        elif time.monotonic() - last_yield_time >= _KEEPALIVE_INTERVAL:
            last_yield_time = time.monotonic()
            yield ('keepalive', None)

    while not status_queue.empty():
        try:
            s = status_queue.get_nowait()
            if s is not None:
                yield ('status', s)
        except asyncio.QueueEmpty:
            break

    await task

    result_type, result_value = await result_queue.get()
    yield (result_type, result_value)


def ask_stream():
    user_id, auth_error = _authenticate_request()
    if auth_error:
        return auth_error

    user_input = request.form.get("text")
    history = request.form.get("history", "")
    _raw = request.form.get("research", "")
    research = _raw is True or (isinstance(_raw, str) and _raw.lower() in ("true", "1", "research"))
    uploaded_file = _extract_upload()

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = _stream_with_status(user_input, history, user_id, research,
                                      uploaded_file=uploaded_file)
            ait = gen.__aiter__()
            while True:
                try:
                    event_type, value = loop.run_until_complete(ait.__anext__())
                except StopAsyncIteration:
                    break

                if event_type == 'keepalive':
                    yield ": keepalive\n\n"
                elif event_type == 'status':
                    yield f"event: status\ndata: {json.dumps({'message': value})}\n\n"
                elif event_type == 'result':
                    messages = value.new_messages()
                    for message in messages:
                        message.parts[:] = [
                            p for p in message.parts
                            if not (isinstance(p, TextPart) and p.content == "")
                        ]
                    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
                    yield f"event: done\ndata: {json.dumps({'response': serialized})}\n\n"
                elif event_type == 'error':
                    logger.error(f"ask_stream agent error: {value}")
                    user_prompt = user_input_to_model_request(user_input)
                    error_response = exception_to_model_response(value)
                    serialized = ModelMessagesTypeAdapter.dump_python(
                        [user_prompt, error_response], mode='json'
                    )
                    yield f"event: done\ndata: {json.dumps({'response': serialized})}\n\n"
        finally:
            loop.close()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


blueprint.add_url_rule(
    "/chat",
    view_func=ChatView.as_view(str("chat")),
    strict_slashes=False,
)

blueprint.add_url_rule(
    "/chat/ask",
    view_func=ask,
    methods=["POST"],
)

blueprint.add_url_rule(
    "/chat/ask/stream",
    view_func=ask_stream,
    methods=["POST"],
)


def get_blueprint():
    return blueprint
