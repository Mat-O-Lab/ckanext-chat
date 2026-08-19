# ckanext-chat

[![Tests](https://github.com/Mat-O-Lab/ckanext-chat/actions/workflows/test.yml/badge.svg)](https://github.com/Mat-O-Lab/ckanext-chat/actions/workflows/test.yml)

A CKAN plugin that adds an AI chat assistant powered by [pydantic-ai](https://ai.pydantic.dev/). The agent can execute CKAN actions with full user-aware authorization, search literature via vector RAG, and analyze documents. Chat histories are stored in browser local storage and passed as context to the agent.

The chat UI renders responses with marked.js and highlight.js. All CKAN operations respect the logged-in user's permissions.

![chat example](./ckanext-chat.PNG)

## Architecture

```
User --> Chat UI (/chat)  --> front_agent / research_agent
                                  |
              +-------------------+-------------------+
              |                   |                   |
         ckan_run /          literature_search   literature_analyse
         mcp_call/mcp_tools       |                   |
              |                rag_agent           doc_agent
              v                   |                   |
      CKAN actions or         Milvus vector       Document text
      MCP JSON-RPC            store (RAG)         extraction
```

**Agents:**
- **front_agent** -- coordinates user requests, delegates to tools (max 3 tool calls)
- **research_agent** -- deep multi-source research with literature search + analysis (max 10 tool calls)
- **ckan_agent** -- validates and optimizes CKAN action calls (used as fallback when MCP unavailable)
- **rag_agent** -- vector search over Milvus for literature retrieval
- **doc_agent** -- targeted document section extraction and analysis

## ckanext-mcp Integration

When [ckanext-mcp](https://github.com/Mat-O-Lab/ckanext-mcp) is installed and loaded, the data fetch layer
uses MCP's JSON-RPC endpoint instead of direct `toolkit.get_action()` calls.

**How the data flow works:**

```
front_agent -> ckan_run -> ckan_agent (validate/correct action + params)
                        -> MCP fetch (preferred) or direct Python (fallback)
                        -> smart_truncate -> return to front_agent
```

The `ckan_agent` LLM validates and optimizes the action (e.g. redirecting `package_list`
to `package_search`, adding `include_private=True`). The actual data fetch goes through MCP
when available, falling back to direct `toolkit.get_action()` calls if MCP is not loaded
or the action is not mapped.

**Why MCP improves the data fetch:**

- **Consistent auth model.** MCP calls go through CKAN's HTTP auth pipeline with per-user API tokens. Permissions are enforced the same way as any API call.
- **Smart defaults and pagination.** MCP applies intelligent parameter defaults at the server level.
- **Supports full CRUD.** Read (search, show, list) and write (create, update, patch) operations are all routed through MCP when available.

**How it works:**

The plugin detects `ckanext-mcp` at request time via `ckan.plugins.plugin_loaded('mcp')`.
If available, it creates a per-user API token and derives the MCP URL from
`ckan.devserver.host` and `ckan.devserver.port` (internal HTTP, no SSL).

To enable, add `mcp` to `ckan.plugins` -- no extra configuration needed.

## OpenAI-Compatible Chat Completions API

The plugin exposes an endpoint compatible with the
[OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).
Any client that speaks this protocol can use it --
the `openai` Python/JS SDKs, `curl`, LangChain, etc.

### Endpoint

```
POST /chat/v1/chat/completions
```

Located on the same CKAN host, e.g.
`https://your-ckan.org/chat/v1/chat/completions`.

### Authentication

CKAN API token in the `Authorization: Bearer <token>` header.
Generate one in the CKAN UI under *User > API Tokens*.

### Compatibility

| Feature | Supported |
|---------|-----------|
| `messages` array (user/assistant/system roles) | yes |
| `stream: false` (default) | yes |
| `stream: true` (SSE) | yes |
| `model` selection | `"default"` or `"research"` |
| `temperature`, `top_p`, `max_tokens` | ignored (agent manages these) |
| `tools` / `functions` | not applicable (agent has its own tools) |
| `n` > 1 (multiple choices) | no |
| Usage stats in response | yes (`prompt_tokens`, `completion_tokens`) |

Streaming follows the SSE format with `chat.completion.chunk` objects
and terminates with `data: [DONE]`.

### Examples

**curl (non-streaming):**
```bash
curl -X POST https://your-ckan.org/chat/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CKAN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What datasets are available?"}
    ]
  }'
```

**curl (streaming):**
```bash
curl -N -X POST https://your-ckan.org/chat/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CKAN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Search for climate data"}
    ],
    "stream": true
  }'
```

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-ckan.org/chat/v1",
    api_key="YOUR_CKAN_API_TOKEN",
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "List all datasets"}
    ],
)
print(response.choices[0].message.content)
```

Set `model` to `"research"` to use the research agent
(deeper multi-source analysis, up to 10 tool calls)
instead of the default front agent (quick answers, up to 3 tool calls).

## LLM Compatibility

OpenAI models from gpt-4o-mini onward work well. Local LLMs tested with ollama:

| LLM | Compatible? |
|-----|-------------|
| qwen2.5:32b | works, some odd output |
| llama3.3:70B | works, reluctant to call tools |
| gemma3 | no tool support |
| phi4 | no tool support |
| qwq | too much reasoning, not enough action |
| mistral:7B | poor tool integration via OpenAI interface |

Reasoning models generally don't perform well -- tool calling needs fast, decisive models.

## Requirements

Compatibility with CKAN versions:

| CKAN version | Compatible? |
|--------------|-------------|
| 2.9 and earlier | not tested |
| 2.10 | yes |
| 2.11 | yes |

## Installation

1. Activate your CKAN virtual environment:
```bash
. /usr/lib/ckan/default/bin/activate
```

2. Install the package:
```bash
pip install ckanext-chat
```

3. Add `chat` to `ckan.plugins` in your CKAN config.

4. Restart CKAN.

## Config Settings

### Model provider (required)

**Azure OpenAI** (default):
```bash
CKANINI__CKANEXT__CHAT__PROVIDER="azure"
CKANINI__CKANEXT__CHAT__BASE_URL="https://your-subscription.openai.azure.com/"
CKANINI__CKANEXT__CHAT__API_KEY="your-api-key"
CKANINI__CKANEXT__CHAT__MODEL_NAME="gpt-4o-mini"
CKANINI__CKANEXT__CHAT__API_VERSION="2024-06-01"
```

**OpenAI-compatible endpoint** (ollama, vLLM, LiteLLM, etc.):
```bash
CKANINI__CKANEXT__CHAT__PROVIDER="openai"
CKANINI__CKANEXT__CHAT__BASE_URL="http://localhost:11434/v1"
CKANINI__CKANEXT__CHAT__API_KEY="ollama"
CKANINI__CKANEXT__CHAT__MODEL_NAME="qwen2.5:32b"
```

**Backward compatibility:** The old config keys (`completion_url`, `api_token`, `deployment`) still work. New keys take precedence when set.

### Optional: thinking model

For the research agent, you can configure a separate model optimized for reasoning:
```bash
CKANINI__CKANEXT__CHAT__THINK_MODEL_NAME="gpt-4.1-mini"
```

Falls back to `model_name` if not set.

### Optional: Milvus RAG

```bash
CKANINI__CKANEXT__CHAT__MILVUS_URL="http://milvus:19530"
CKANINI__CKANEXT__CHAT__COLLECTION_NAME="documents"
CKANINI__CKANEXT__CHAT__EMBEDDING_MODEL="text-embedding-3-small"
CKANINI__CKANEXT__CHAT__EMBEDDING_API="https://your-embedding-api"
```

Without these, the literature search agent relies on `package_search`.

### All config keys

| Key | Default | Description |
|-----|---------|-------------|
| `ckanext.chat.provider` | `azure` | `azure` or `openai` |
| `ckanext.chat.base_url` | | Model provider endpoint URL |
| `ckanext.chat.api_key` | | API key for the model provider |
| `ckanext.chat.model_name` | `gpt-4o-mini` | Model for front_agent, ckan_agent, rag_agent, doc_agent |
| `ckanext.chat.think_model_name` | | Model for research_agent (falls back to model_name) |
| `ckanext.chat.api_version` | `2024-06-01` | Azure API version |
| `ckanext.chat.milvus_url` | | Milvus vector store URL |
| `ckanext.chat.collection_name` | | Milvus collection name |
| `ckanext.chat.embedding_model` | `text-embedding-3-small` | Embedding model name |
| `ckanext.chat.embedding_api` | | Embedding API endpoint |
| `ckanext.chat.ssl_verify` | `true` | Verify SSL certificates for resource downloads |
| `ckanext.chat.completion_url` | | (legacy) Azure endpoint, use `base_url` instead |
| `ckanext.chat.api_token` | | (legacy) API key, use `api_key` instead |
| `ckanext.chat.deployment` | `gpt-4o-mini` | (legacy) Model name, use `model_name` instead |

## Timeouts

Proxy in front of CKAN must allow long-running API calls. For nginx:
```conf
proxy_connect_timeout 3600s;
proxy_read_timeout 3600s;
proxy_send_timeout 3000s;
send_timeout 3000;
```

For production with uWSGI, set harakiri in `start_ckan.sh`:
```bash
UWSGI_OPTS="--socket /tmp/uwsgi.sock \
            --wsgi-file /srv/app/wsgi.py \
            --module wsgi:application \
            --http 0.0.0.0:5000 \
            --master --enable-threads \
            --lazy-apps \
            -p 2 -L -b 32768 --vacuum \
            --harakiri-verbose \
            --socket-timeout $UWSGI_HARAKIRI \
            --harakiri $UWSGI_HARAKIRI \
            --http-timeout $UWSGI_HARAKIRI"
```
```bash
UWSGI_HARAKIRI="3000"
```

## Round-Trip Integration Test

A live integration test that drives CKAN CRUD operations through the chat completions
endpoint using natural language. Analogous to ckanext-mcp's `test_crud_lifecycle.py`
but LLM-driven.

The test creates an org, dataset, resource, patches metadata, searches, and cleans up:

```bash
# With API token
python ckanext/chat/tests/test_chat_roundtrip.py \
  --url http://localhost:80 \
  --token YOUR_CKAN_API_TOKEN \
  --verbose

# With username/password (creates its own token)
python ckanext/chat/tests/test_chat_roundtrip.py \
  --url http://localhost:80 \
  --user ckan_admin --password your_password \
  --verbose

# Cleanup leftover test artifacts from previous runs
python ckanext/chat/tests/test_chat_roundtrip.py \
  --url http://localhost:80 --token YOUR_TOKEN --cleanup
```

Since responses depend on LLM behavior, some verification checks may fail
even when the underlying CKAN operations succeed. The CRUD steps (create, patch)
are the primary validation; search/show checks verify response quality.

## Developer Installation

```bash
git clone https://github.com/Mat-O-Lab/ckanext-chat.git
cd ckanext-chat
pip install -e ".[dev]"
```

## Tests

```bash
pytest --ckan-ini=test.ini
```

## AI-Assisted Documentation

Parts of this repository were created with the assistance of artificial intelligence (AI) tools. While the content has been reviewed and edited for accuracy, users should:

- Verify critical information independently
- Report any errors or inconsistencies via the GitHub issue tracker
- Exercise due diligence when following instructions
- The use of AI does not affect the functionality or reliability of the software itself.



## License

[AGPL](https://www.gnu.org/licenses/agpl-3.0.en.html)

