"""
Azure OpenAI Weather Agent with Full OpenTelemetry Support
Sends traces, metrics, and logs to SigNoz
"""

import os
import re
import time
import logging
import gradio as gr
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# =============================================================================
# OpenTelemetry Setup - Traces, Metrics, and Logs
# =============================================================================
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.trace import Status, StatusCode

# Logging with OTEL
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# Configuration
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "azure-openai-weather-agent")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

# Resource identifies this service
resource = Resource.create({
    "service.name": SERVICE_NAME,
    "service.version": "1.0.0",
    "deployment.environment": "demo",
    "cloud.provider": "azure",
})

# -----------------------------------------------------------------------------
# 1. TRACING SETUP
# -----------------------------------------------------------------------------
tracer_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("azure-openai-weather-agent", "1.0.0")

# -----------------------------------------------------------------------------
# 2. METRICS SETUP
# -----------------------------------------------------------------------------
metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)
metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("azure-openai-weather-agent", "1.0.0")

# Create metric instruments
token_counter = meter.create_counter("agent.tokens.total", description="Total tokens used", unit="tokens")
input_token_counter = meter.create_counter("agent.tokens.input", description="Input tokens", unit="tokens")
output_token_counter = meter.create_counter("agent.tokens.output", description="Output tokens", unit="tokens")
request_latency = meter.create_histogram("agent.request.latency", description="Request latency", unit="ms")
tool_calls_counter = meter.create_counter("agent.tool.calls", description="Tool calls", unit="calls")
error_counter = meter.create_counter("agent.errors", description="Errors", unit="errors")

def record_token_usage(response, span):
    """Extract and record token usage from LangChain response."""
    try:
        # Token usage is in response_metadata for Azure OpenAI
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            usage = metadata.get('token_usage', {}) or metadata.get('usage', {})
            
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
            
            if total_tokens > 0:
                # Record metrics
                input_token_counter.add(input_tokens, {"model": "gpt-5.2"})
                output_token_counter.add(output_tokens, {"model": "gpt-5.2"})
                token_counter.add(total_tokens, {"model": "gpt-5.2"})
                
                # Record in span
                span.set_attribute("llm.tokens.input", input_tokens)
                span.set_attribute("llm.tokens.output", output_tokens)
                span.set_attribute("llm.tokens.total", total_tokens)
                
                logger.info(f"Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                return total_tokens
    except Exception as e:
        logger.warning(f"Could not extract token usage: {e}")
    return 0

# -----------------------------------------------------------------------------
# 3. LOGGING SETUP
# -----------------------------------------------------------------------------
logger_provider = LoggerProvider(resource=resource)
log_exporter = OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=True)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
set_logger_provider(logger_provider)

# Create Python logger with OTEL handler
otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
logger = logging.getLogger("weather-agent")
logger.setLevel(logging.INFO)
logger.addHandler(otel_handler)
# Also log to console
logger.addHandler(logging.StreamHandler())

# =============================================================================
# Sensitive Data Redaction
# =============================================================================
SENSITIVE_PATTERNS = [
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***REDACTED***'),
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***REDACTED***'),
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), r'\1=***REDACTED***'),
]

def redact(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# =============================================================================
# Weather Tools
# =============================================================================
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "new york": "18°C, Clear, Breezy",
        "london": "12°C, Overcast, Light Drizzle",
        "tokyo": "22°C, Sunny, Humid",
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    return f"Weather not available for {city}. Try: {', '.join(weather_data.keys())}"

@tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a city."""
    forecasts = {
        "mumbai": ["32°C Sunny", "31°C Cloudy", "30°C Rain"],
        "delhi": ["28°C Clear", "30°C Sunny", "29°C Hazy"],
        "bangalore": ["24°C Rain", "23°C Cloudy", "25°C Pleasant"],
        "new york": ["17°C Sunny", "19°C Clear", "16°C Cloudy"],
        "london": ["11°C Rain", "10°C Overcast", "12°C Cloudy"],
    }
    city_lower = city.lower().strip()
    if city_lower in forecasts:
        forecast_list = forecasts[city_lower][:min(days, 3)]
        return f"Forecast for {city.title()}: " + ", ".join([f"Day {i+1}: {f}" for i, f in enumerate(forecast_list)])
    return f"Forecast not available for {city}."

tools = [get_weather, get_forecast]

SYSTEM_PROMPT = """You are a weather assistant. Use get_weather for current conditions and get_forecast for predictions.
Available cities: Mumbai, Delhi, Bangalore, New York, London, Tokyo. Be concise."""

# =============================================================================
# LLM Setup
# =============================================================================
def get_llm():
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not azure_api_key:
        return None
    
    return AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sre-resources.openai.azure.com"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        api_key=azure_api_key,
    ).bind_tools(tools)

# =============================================================================
# Chat Function with Full OTEL Instrumentation
# =============================================================================
def chat(message, history):
    llm = get_llm()
    if not llm:
        return "❌ AZURE_OPENAI_API_KEY not set"
    
    # Start parent span for entire request
    with tracer.start_as_current_span("agent_request") as span:
        start_time = time.time()
        span.set_attribute("user.input", message[:200])
        span.set_attribute("llm.provider", "azure_openai")
        span.set_attribute("llm.model", "gpt-5.2")
        
        logger.info(f"Processing request: {message[:100]}...")
        
        # Build messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for h in history:
            messages.append(HumanMessage(content=h[0]))
            if h[1]:
                messages.append(AIMessage(content=h[1]))
        messages.append(HumanMessage(content=message))
        
        try:
            # LLM call with span
            with tracer.start_as_current_span("llm_call") as llm_span:
                llm_start = time.time()
                llm_span.set_attribute("llm.model", "gpt-5.2")
                
                # Record request (input messages)
                request_text = "\n".join([f"{m.type}: {m.content[:200]}" for m in messages])
                llm_span.set_attribute("llm.request", request_text[:1000])
                
                response = llm.invoke(messages)
                
                llm_duration = (time.time() - llm_start) * 1000
                llm_span.set_attribute("llm.duration_ms", llm_duration)
                
                # Record response
                if response.tool_calls:
                    llm_span.set_attribute("llm.response", f"Tool calls: {[tc['name'] for tc in response.tool_calls]}")
                else:
                    llm_span.set_attribute("llm.response", response.content[:500])
                
                record_token_usage(response, llm_span)
                llm_span.set_status(Status(StatusCode.OK))
                logger.info(f"LLM call completed in {llm_duration:.0f}ms")
            
            # Handle tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    with tracer.start_as_current_span(f"tool_{tool_name}") as tool_span:
                        tool_span.set_attribute("tool.name", tool_name)
                        tool_span.set_attribute("tool.args", str(tool_args))
                        tool_calls_counter.add(1, {"tool": tool_name})
                        logger.info(f"Calling tool: {tool_name}")
                        
                        if tool_name == "get_weather":
                            result = get_weather.invoke(tool_args)
                        elif tool_name == "get_forecast":
                            result = get_forecast.invoke(tool_args)
                        else:
                            result = "Unknown tool"
                        
                        tool_span.set_attribute("tool.result", result[:200])
                        tool_span.set_status(Status(StatusCode.OK))
                        
                        messages.append(response)
                        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                
                # Final LLM call after tools
                with tracer.start_as_current_span("llm_call_final") as llm_span:
                    llm_span.set_attribute("llm.model", "gpt-5.2")
                    
                    # Record request with tool results
                    request_text = "\n".join([f"{m.type}: {str(m.content)[:200]}" for m in messages[-3:]])
                    llm_span.set_attribute("llm.request", request_text[:1000])
                    
                    response = llm.invoke(messages)
                    
                    # Record response
                    llm_span.set_attribute("llm.response", response.content[:500])
                    
                    record_token_usage(response, llm_span)
                    llm_span.set_status(Status(StatusCode.OK))
            
            # Record success
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.response", response.content[:200])
            span.set_status(Status(StatusCode.OK))
            
            request_latency.record(duration_ms, {"status": "success"})
            logger.info(f"Request completed in {duration_ms:.0f}ms")
            
            return response.content
            
        except Exception as e:
            safe_error = redact(str(e))
            span.set_status(Status(StatusCode.ERROR, safe_error))
            span.set_attribute("error.message", safe_error)
            error_counter.add(1, {"type": type(e).__name__})
            logger.error(f"Error: {safe_error}")
            return f"❌ Error: {safe_error}"

# =============================================================================
# Gradio UI
# =============================================================================
demo = gr.ChatInterface(
    fn=chat,
    title="🌤️ Demo Weather Agent with O11Y",
    description="Ask about weather. Traces, metrics, and logs are sent to SigNoz at http://localhost:8080",
    examples=["What's the weather in Mumbai?", "Forecast for New York"],
)

if __name__ == "__main__":
    logger.info("Starting Azure OpenAI Weather Agent with OTEL")
    logger.info(f"OTLP Endpoint: {OTLP_ENDPOINT}")
    logger.info(f"Service Name: {SERVICE_NAME}")
    demo.launch(server_port=8000)
