# Demo: Weather Agent WITH OpenTelemetry

This demo shows the same weather agent, but now instrumented with **full OpenTelemetry observability** - the "glass box" approach.

## What This Demo Shows

- The same weather agent from `demo-without-otel`
- **Full visibility** into internal operations via OpenTelemetry
- Traces, metrics, and logs sent to SigNoz

## What You Can Now See

| Signal | What's Captured |
|--------|-----------------|
| **Traces** | Full request flow with spans for LLM calls and tool calls |
| **Metrics** | Token usage, latency histograms, tool call counts, errors |
| **Logs** | Tool invocations, LLM completions, errors with context |

### Span Attributes Captured

- `llm.model` - Model name (gemini-1.5-flash)
- `llm.tokens.input` - Input tokens per LLM call
- `llm.tokens.output` - Output tokens per LLM call
- `llm.tokens.total` - Total tokens per LLM call
- `llm.duration_ms` - LLM call latency
- `tool.name` - Tool function name
- `tool.inputs` - Tool input parameters
- `tool.output` - Tool return value
- `tool.duration_ms` - Tool execution time

### Metrics Available

- `agent.tokens.input` - Counter of input tokens
- `agent.tokens.output` - Counter of output tokens
- `agent.request.latency` - Histogram of request latency (ms)
- `agent.tool.calls` - Counter of tool calls by tool name
- `agent.errors` - Counter of errors by type

## Prerequisites

1. **SigNoz running locally** (for receiving OTEL data)

### Quick SigNoz Setup (if not already running)

```bash
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker-compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

SigNoz UI will be at: http://localhost:3301

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install google-adk google-generativeai litellm
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
```

### 3. Set Environment Variables

```bash
# Required: Gemini API key
export GOOGLE_API_KEY="your-gemini-api-key"

# OTEL Configuration (defaults shown - adjust if needed)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="weather-agent-otel"
```

## Running the Demo

### 1. Start the ADK Web Interface

```bash
cd demo-with-otel
adk web
```

### 2. Open the Chat Interface

Open your browser to: **http://localhost:8000**

### 3. Interact with the Agent

Try these queries:
1. "What's the weather in Mumbai?"
2. "Give me a 5-day forecast for Bangalore"
3. "Compare weather in Delhi and Chennai"

### 4. View Observability Data in SigNoz

Open: **http://localhost:3301**

#### Traces
1. Go to **Traces** tab
2. Filter by service: `weather-agent-otel`
3. Click on a trace to see the span hierarchy
4. Check span attributes for token counts

#### Metrics
1. Go to **Dashboards** or **Metrics Explorer**
2. Search for metrics starting with `agent.`
3. Create graphs for token usage, latency, etc.

#### Logs
1. Go to **Logs** tab
2. Filter by service name
3. See tool calls and LLM completions

## Project Structure

```
demo-with-otel/
├── README.md              # This file
└── weather_agent/
    ├── __init__.py        # Package marker
    ├── otel_setup.py      # OpenTelemetry initialization
    └── agent.py           # Instrumented agent with tools
```

## Key Files Explained

### `otel_setup.py`
- Initializes OpenTelemetry tracer and meter
- Configures OTLP exporters for SigNoz
- Creates metric instruments (counters, histograms)
- Runs at import time so OTEL is ready before ADK starts

### `agent.py`
- `@instrumented_tool` decorator - Wraps tools with spans and metrics
- `InstrumentedLiteLlm` - LLM wrapper that captures token usage
- Same agent logic as demo-without-otel, but now observable

## Comparing Black Box vs Glass Box

| Aspect | demo-without-otel | demo-with-otel |
|--------|-------------------|----------------|
| Token visibility | ❌ None | ✅ Per-span and total |
| Latency tracking | ❌ None | ✅ Histograms with percentiles |
| Tool call tracking | ❌ None | ✅ Counts, inputs, outputs |
| Error tracking | ❌ None | ✅ Counts by type, stack traces |
| Request flow | ❌ None | ✅ Full trace with spans |
| Debugging | 🔴 Guessing | 🟢 Data-driven |

## Troubleshooting

### No data in SigNoz?

1. Check SigNoz is running: `docker ps | grep signoz`
2. Verify OTLP endpoint: `echo $OTEL_EXPORTER_OTLP_ENDPOINT`
3. Check for connection errors in agent logs

### Token counts showing 0?

Token extraction depends on the LLM provider's response format.
The current implementation works with standard OpenAI-compatible responses.
Gemini via LiteLLM may have different response structures.

### Spans not appearing?

Spans are batched and sent periodically. Wait 10-15 seconds after
making requests, then refresh SigNoz.
