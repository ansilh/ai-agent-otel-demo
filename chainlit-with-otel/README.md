# Demo: Weather Agent using Chainlit WITH OpenTelemetry

This demo shows the same weather agent, but now instrumented with **full OpenTelemetry observability** - the "glass box" approach.

## Why Chainlit + OTEL?

| Feature | Benefit |
|---------|---------|
| **Beautiful UI** | Modern chat interface with thinking indicators |
| **Full tracing** | See every LLM call and tool invocation |
| **Metrics** | Token usage, latency histograms |
| **Easy integration** | OTEL setup in ~50 lines |

## What You Can Now See

| Signal | What's Captured |
|--------|-----------------|
| **Traces** | Full request flow with spans for LLM calls and tool calls |
| **Metrics** | Token usage, latency histograms, tool call counts |
| **Span Attributes** | Model, tokens, duration, inputs, outputs |

## Prerequisites

1. **Ollama** running locally (for Qwen model)
2. **SigNoz** running locally (for OTEL data)

### Quick SigNoz Setup

```bash
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker-compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

SigNoz UI: http://localhost:3301

## Setup

### 1. Install and Start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen model
ollama pull qwen2.5:7b

# Start Ollama server
ollama serve
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd chainlit-with-otel
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)

```bash
# Defaults shown - adjust if needed
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="chainlit-weather-agent"
```

## Running the Demo

```bash
chainlit run app.py
```

- **Chat UI**: http://localhost:8000
- **SigNoz**: http://localhost:3301

## Try These Queries

1. "What's the weather in Mumbai?"
2. "Give me a 5-day forecast for Bangalore"
3. "Compare weather in Delhi and Chennai"

Then check SigNoz to see:
- **Traces** → Search for `chainlit-weather-agent`
- **Metrics** → Look for `agent.*` metrics

## Project Structure

```
chainlit-with-otel/
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── app.py              # Main application with OTEL
```

## Code Highlights

### OTEL Setup (~20 lines)
```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
# ... setup tracer and meter
tracer = trace.get_tracer("chainlit-weather-agent")
```

### Instrumented Tool
```python
@tool
@instrumented_tool  # Adds OTEL tracing
def get_weather(city: str) -> str:
    ...
```

### Traced Request Handler
```python
@cl.on_message
async def main(message: cl.Message):
    with tracer.start_as_current_span("agent_request") as span:
        span.set_attribute("user.input", message.content)
        # ... run agent
```

## Comparing Black Box vs Glass Box

| Aspect | chainlit-without-otel | chainlit-with-otel |
|--------|----------------------|-------------------|
| Token visibility | ❌ None | ✅ Per-call and total |
| Latency tracking | ❌ None | ✅ Histograms |
| Tool call tracking | ❌ None | ✅ Counts, inputs, outputs |
| Error tracking | ❌ None | ✅ Stack traces in spans |
| Request flow | ❌ None | ✅ Full trace hierarchy |
