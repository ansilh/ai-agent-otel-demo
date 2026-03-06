# Demo: Weather Agent using Azure OpenAI with OpenTelemetry

This demo uses **vendor-neutral, open-source SDKs** to build an observable AI agent powered by Azure OpenAI.

## Tech Stack (All Open Source / Vendor Neutral)

| Component | Technology | Why |
|-----------|------------|-----|
| **Agent Framework** | LangChain | Open source, works with any LLM provider |
| **Web UI** | Chainlit | Open source, beautiful chat interface |
| **Observability** | OpenTelemetry | CNCF standard, vendor neutral |
| **LLM Provider** | Azure OpenAI | Enterprise-grade, GPT-5.2 |

## What You Can See

| Signal | What's Captured |
|--------|-----------------|
| **Traces** | Full request flow with spans for LLM calls and tool calls |
| **Metrics** | Token usage (input/output/total), latency histograms |
| **Span Attributes** | Provider, model, tokens, duration, inputs, outputs |

## Prerequisites

1. **Azure OpenAI** access with GPT-5.2 deployment
2. **SigNoz** running locally (for OTEL data)

### Quick SigNoz Setup

```bash
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker-compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

SigNoz UI: http://localhost:3301

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
cd azure-openai-demo
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
# Required: Azure OpenAI credentials
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="https://sre-resources.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-5.2"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Optional: OTEL configuration (defaults shown)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="azure-openai-weather-agent"
```

### 4. Verify Azure OpenAI Access

Make sure your Azure OpenAI resource has:
- Endpoint: `https://sre-resources.openai.azure.com`
- Deployment name: `gpt-5.2`
- API version: `2025-04-01-preview`

## Running the Demo

```bash
chainlit run app.py
```

- **Chat UI**: http://localhost:8000
- **SigNoz**: http://localhost:3301

## Testing the Agent

### Try These Queries

1. **Current weather**:
   - "What's the weather in Mumbai?"
   - "How's the weather in New York?"

2. **Forecasts**:
   - "Give me a 5-day forecast for Bangalore"
   - "What's the weather forecast for London?"

3. **Comparisons**:
   - "Compare weather in Delhi and Tokyo"

### Available Cities

- **India**: Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune
- **International**: New York, London, Tokyo

## Viewing Observability Data

### In SigNoz (http://localhost:3301)

#### Traces
1. Go to **Traces** tab
2. Filter by service: `azure-openai-weather-agent`
3. Click on a trace to see:
   - `agent_request` (parent span)
   - `llm_call` (Azure OpenAI calls)
   - `tool_get_weather` / `tool_get_forecast` (tool calls)

#### Metrics
1. Go to **Dashboards** or **Metrics Explorer**
2. Search for metrics:
   - `agent.tokens.total` - Total tokens used
   - `agent.tokens.input` - Prompt tokens
   - `agent.tokens.output` - Completion tokens
   - `agent.request.latency` - Request latency
   - `agent.tool.calls` - Tool call counts

### Span Attributes

Each trace includes these attributes:
- `llm.provider`: `azure_openai`
- `llm.model`: `gpt-5.2`
- `llm.tokens.input`: Input token count
- `llm.tokens.output`: Output token count
- `llm.tokens.total`: Total token count
- `llm.duration_ms`: LLM call latency
- `tool.name`: Tool function name
- `tool.inputs`: Tool input parameters
- `tool.output`: Tool return value

## Project Structure

```
azure-openai-demo/
├── README.md           # This file
├── requirements.txt    # Python dependencies (all open source)
└── app.py              # Main application
```

## Code Highlights

### Vendor-Neutral LLM Setup
```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_version=azure_api_version,
    api_key=azure_api_key,
)
```

### Open Source Observability
```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Works with any OTEL-compatible backend (SigNoz, Jaeger, Grafana, etc.)
tracer = trace.get_tracer("azure-openai-weather-agent")
```

### Instrumented Tools
```python
@tool
@instrumented_tool  # Adds OTEL tracing
def get_weather(city: str) -> str:
    ...
```

## Switching LLM Providers

Because we use vendor-neutral SDKs, you can easily switch providers:

### To use OpenAI directly:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", api_key=os.environ["OPENAI_API_KEY"])
```

### To use Ollama (local):
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b")
```

### To use Anthropic:
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet")
```

The observability code remains unchanged!

## Troubleshooting

### "AZURE_OPENAI_API_KEY not set"
Make sure to export the environment variable:
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
```

### "Deployment not found"
Verify your Azure OpenAI deployment name matches `AZURE_OPENAI_DEPLOYMENT`.

### No data in SigNoz?
1. Check SigNoz is running: `docker ps | grep signoz`
2. Verify OTLP endpoint: `echo $OTEL_EXPORTER_OTLP_ENDPOINT`
3. Wait 10-15 seconds for data to appear (spans are batched)

### Rate limiting errors
Azure OpenAI has rate limits. If you hit them:
- Wait a few seconds between requests
- Check your Azure OpenAI quota in Azure Portal
