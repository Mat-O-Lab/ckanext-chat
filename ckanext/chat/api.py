import asyncio
import json
import os
import time
import uuid
from distutils.util import strtobool
from typing import Any

import ckan.lib.api_token as api_token
import ckan.plugins.toolkit as toolkit
from flask import Blueprint, Response, jsonify, request, stream_with_context
from loguru import logger
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import UsageLimits

api_blueprint = Blueprint("chat_api", __name__)

log = logger.bind(module=__name__)

_SSE_KEEPALIVE = "\x00keepalive"
_KEEPALIVE_INTERVAL = 15


@api_blueprint.before_request
def _capture_app():
    from ckanext.chat import views
    if views.global_ckan_app is None:
        from flask import current_app
        views.global_ckan_app = current_app._get_current_object()


def _authenticate():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None, _error_response("Missing Authorization header", 401, "invalid_api_key")

    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        return None, _error_response("Invalid Authorization header", 401, "invalid_api_key")

    user = api_token.get_user_from_token(token)
    if not user:
        return None, _error_response("Invalid API token", 401, "invalid_api_key")

    return user, None


def _error_response(message: str, status: int = 400, error_type: str = "invalid_request_error", param: str = None):
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }), status


def _openai_messages_to_prompt(messages: list) -> tuple[str, list]:
    prompt = ""
    history_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            prompt = content
        elif role == "assistant":
            history_parts.append(
                {"kind": "request", "parts": [{"type": "user-prompt", "content": prompt, "timestamp": "2024-01-01T00:00:00Z"}]}
            )
            history_parts.append(
                {"kind": "response", "parts": [{"type": "text", "content": content}], "model_name": "agent", "timestamp": "2024-01-01T00:00:00Z"}
            )
            prompt = ""
        elif role == "system":
            pass

    if not prompt and messages:
        last = messages[-1]
        if last.get("role") == "user":
            prompt = last.get("content", "")

    return prompt, history_parts


def _setup_agent_run(user_id: str, history_parts: list, research: bool):
    from ckanext.chat.bot.agent import (
        Deps, agent, research_agent,
        mcp_available, get_user_token, config,
    )
    from ckanext.chat.bot.utils import init_dynamic_models, dynamic_models_initialized

    if not dynamic_models_initialized:
        init_dynamic_models()

    deps = Deps(user_id=user_id)

    if mcp_available():
        host = toolkit.config.get("ckan.devserver.host", "localhost")
        port = toolkit.config.get("ckan.devserver.port", "5000")
        mcp_url = f"http://{host}:{port}/mcp"
        token = get_user_token(user_id)
        if token:
            deps.mcp_token = token
            deps.mcp_url = mcp_url

    msg_history = None
    if history_parts:
        try:
            msg_history = ModelMessagesTypeAdapter.validate_python(history_parts)
        except Exception:
            msg_history = None

    active_agent = research_agent if research else agent
    if research:
        log.info("Switching from front_agent to research_agent")
    limits = (
        UsageLimits(request_limit=config.REQUEST_LIMIT_RESEARCH_AGENT, total_tokens_limit=config.MAX_TOKENS_RESEARCH_AGENT)
        if research else
        UsageLimits(request_limit=config.REQUEST_LIMIT_FRONT_AGENT, total_tokens_limit=config.MAX_TOKENS_FRONT_AGENT)
    )
    return active_agent, deps, msg_history, limits


async def _run_agent_for_api(prompt: str, history_parts: list, user_id: str, research: bool = False) -> Any:
    active_agent, deps, msg_history, limits = _setup_agent_run(user_id, history_parts, research)
    return await active_agent.run(
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
        usage_limits=limits,
    )


async def _run_agent_stream(prompt: str, history_parts: list, user_id: str, research: bool = False):
    active_agent, deps, msg_history, limits = _setup_agent_run(user_id, history_parts, research)

    status_queue = asyncio.Queue()
    deps.status_queue = status_queue
    if research:
        status_queue.put_nowait("Switching to research agent (deep research mode)")
    output_queue = asyncio.Queue()

    t0 = time.monotonic()
    last_yield_time = t0
    first_chunk = True
    chunk_count = 0

    async def _agent_worker():
        try:
            if research:
                # Research agent does many multi-turn tool calls; run_stream()
                # + stream_text() ends prematurely after the first model
                # response text. Use run() (like /chat/ask/stream) instead.
                result = await active_agent.run(
                    user_prompt=prompt,
                    message_history=msg_history,
                    deps=deps,
                    usage_limits=limits,
                )
                text = result.output if hasattr(result, "output") else str(result)
                await output_queue.put(text)
            else:
                async with active_agent.run_stream(
                    user_prompt=prompt,
                    message_history=msg_history,
                    deps=deps,
                    usage_limits=limits,
                ) as stream:
                    async for chunk in stream.stream_text(delta=True):
                        await output_queue.put(chunk)
        except Exception as e:
            log.error(f"agent_worker error: {type(e).__name__}: {str(e)[:500]}")
            await output_queue.put(f"\n\n**Fehler:** {type(e).__name__}: {str(e)}")
        finally:
            await output_queue.put(None)

    task = asyncio.create_task(_agent_worker())

    def _drain_status():
        chunks = []
        while not status_queue.empty():
            try:
                chunks.append(status_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return chunks

    running = True
    while running:
        yielded = False
        for status in _drain_status():
            yield f"[status]{status}[/status]\n"
            yielded = True

        try:
            chunk = await asyncio.wait_for(output_queue.get(), timeout=0.2)
            if chunk is None:
                running = False
            else:
                chunk_count += 1
                if first_chunk:
                    log.info(f"stream first-chunk after {time.monotonic() - t0:.1f}s")
                    first_chunk = False
                yield chunk
                yielded = True
        except asyncio.TimeoutError:
            if task.done():
                running = False

        if yielded:
            last_yield_time = time.monotonic()
        elif time.monotonic() - last_yield_time >= _KEEPALIVE_INTERVAL:
            last_yield_time = time.monotonic()
            yield _SSE_KEEPALIVE

    for status in _drain_status():
        yield f"[status]{status}[/status]\n"

    await task
    log.info(f"stream finished: {chunk_count} chunks in {time.monotonic() - t0:.1f}s")


@api_blueprint.route("/chat/v1/chat/completions", methods=["POST"])
def chat_completions():
    user, err = _authenticate()
    if err:
        return err

    try:
        body = request.get_json(force=True)
    except Exception:
        return _error_response("Invalid JSON body", 400)

    if not body:
        return _error_response("Empty request body", 400)

    messages = body.get("messages", [])
    if not messages:
        return _error_response("messages array is required and must not be empty", 400, param="messages")

    model_hint = body.get("model", "default")
    research = model_hint.lower() in ("research", "research_agent")
    stream = body.get("stream", False)

    prompt, history_parts = _openai_messages_to_prompt(messages)
    if not prompt:
        return _error_response("No user message found in messages array", 400, param="messages")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    debug = bool(strtobool(os.environ.get("DEBUG", "false")))

    if stream:
        return _handle_stream(prompt, history_parts, user, research, completion_id, created, model_hint, debug)
    else:
        return _handle_non_stream(prompt, history_parts, user, research, completion_id, created, model_hint, debug)


def _handle_non_stream(prompt, history_parts, user, research, completion_id, created, model_hint, debug):
    try:
        result = asyncio.run(
            _run_agent_for_api(prompt, history_parts, user.id, research=research),
            debug=debug,
        )
        content = result.output if hasattr(result, "output") else str(result)
        usage = result.usage() if hasattr(result, "usage") else None

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_hint,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": usage.request_tokens if usage else 0,
                "completion_tokens": usage.response_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }
        return jsonify(response)

    except Exception as e:
        log.error(f"chat_completions error: {type(e).__name__}: {str(e)[:200]}")
        return _error_response(f"Agent error: {type(e).__name__}: {str(e)}", 500, "server_error")


def _handle_stream(prompt, history_parts, user, research, completion_id, created, model_hint, debug):
    def generate():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stream_gen = _run_agent_stream(prompt, history_parts, user.id, research=research)
                ait = stream_gen.__aiter__()
                while True:
                    try:
                        chunk_text = loop.run_until_complete(ait.__anext__())
                    except StopAsyncIteration:
                        break

                    if chunk_text == _SSE_KEEPALIVE:
                        yield ": keepalive\n\n"
                        continue

                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_hint,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_hint,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            finally:
                loop.close()

        except Exception as e:
            log.error(f"chat_completions stream error: {type(e).__name__}: {str(e)[:200]}")
            error_msg = f"Stream error: {type(e).__name__}: {str(e)}"
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_hint,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\n**Fehler:** {error_msg}"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_hint,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def get_api_blueprint():
    return api_blueprint
