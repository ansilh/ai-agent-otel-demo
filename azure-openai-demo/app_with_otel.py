"""
=============================================================================
DEMO: Weather Agent with Full OpenTelemetry (O11Y) Support
=============================================================================

This demo shows how to instrument an AI agent with OpenTelemetry to send:
- TRACES: Track request flow through the system (spans)
- METRICS: Numerical measurements (token counts, latency, errors)
- LOGS: Structured log messages

All telemetry is sent to SigNoz via OTLP (OpenTelemetry Protocol).

Run with: python3 app_with_otel.py
UI: http://localhost:8000
SigNoz: http://localhost:8080
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os          # For reading environment variables (API keys, endpoints)
import re          # For regex-based sensitive data redaction
import time        # For measuring request latency
import logging     # Python's built-in logging framework

# Gradio - Web UI framework for creating chat interfaces
import gradio as gr

# LangChain - Framework for building LLM applications
from langchain_openai import AzureChatOpenAI  # Azure OpenAI client
from langchain_core.tools import tool          # Decorator to create tools
from langchain_core.messages import (          # Message types for chat
    HumanMessage,    # User's message
    AIMessage,       # Assistant's response
    SystemMessage,   # System prompt
    ToolMessage,     # Tool execution result
)

# =============================================================================
# OPENTELEMETRY IMPORTS
# =============================================================================

# Core OTEL APIs - these are the interfaces we use
from opentelemetry import trace, metrics

# SDK implementations - these provide the actual functionality
from opentelemetry.sdk.trace import TracerProvider          # Manages trace creation
from opentelemetry.sdk.metrics import MeterProvider         # Manages metric creation
from opentelemetry.sdk.resources import Resource            # Identifies this service

# OTLP Exporters - send telemetry to backends like SigNoz via gRPC
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Processors - batch and export telemetry data
from opentelemetry.sdk.trace.export import BatchSpanProcessor           # Batches spans before export
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # Exports metrics periodically

# Status codes for spans - indicate success or failure
from opentelemetry.trace import Status, StatusCode

# OTEL Logging - sends Python logs to OTEL backend
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Service name appears in SigNoz to identify this application
# Can be overridden via environment variable
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "azure-openai-weather-agent")

# OTLP endpoint - where to send telemetry data
# Default is localhost:4317 (SigNoz OTEL collector gRPC port)
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

# =============================================================================
# RESOURCE DEFINITION
# =============================================================================
# A Resource represents the entity producing telemetry (this service)
# These attributes appear on all traces, metrics, and logs

resource = Resource.create({
    "service.name": SERVICE_NAME,           # Name shown in SigNoz
    "service.version": "1.0.0",             # Version of this service
    "deployment.environment": "demo",        # Environment (dev/staging/prod)
    "cloud.provider": "azure",               # Cloud provider being used
})

# =============================================================================
# 1. TRACING SETUP
# =============================================================================
# Traces show the flow of a request through the system as a tree of "spans"
# Each span represents a unit of work (LLM call, tool execution, etc.)

# TracerProvider is the entry point for creating traces
tracer_provider = TracerProvider(resource=resource)

# OTLPSpanExporter sends spans to SigNoz via gRPC
# insecure=True means no TLS (fine for localhost)
trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)

# BatchSpanProcessor batches spans before sending (more efficient than sending one by one)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

# Register our tracer provider as the global default
trace.set_tracer_provider(tracer_provider)

# Get a tracer instance - we use this to create spans
tracer = trace.get_tracer("weather-agent", "1.0.0")

# =============================================================================
# 2. METRICS SETUP
# =============================================================================
# Metrics are numerical measurements aggregated over time
# Types: Counter (cumulative), Histogram (distribution), Gauge (current value)

# OTLPMetricExporter sends metrics to SigNoz via gRPC
metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)

# PeriodicExportingMetricReader exports metrics every 5 seconds
metric_reader = PeriodicExportingMetricReader(
    metric_exporter, 
    export_interval_millis=5000  # Export every 5 seconds
)

# MeterProvider manages metric instruments
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

# Register as global default
metrics.set_meter_provider(meter_provider)

# Get a meter instance - we use this to create metric instruments
meter = metrics.get_meter("weather-agent", "1.0.0")

# -----------------------------------------------------------------------------
# CREATE METRIC INSTRUMENTS
# -----------------------------------------------------------------------------
# Counters: Cumulative values that only go up (total tokens, total errors)
# Histograms: Distribution of values (latency percentiles)

# Token counters - track LLM token usage for cost monitoring
token_counter = meter.create_counter(
    "agent.tokens.total",           # Metric name (appears in SigNoz)
    description="Total tokens used",
    unit="tokens"
)
input_token_counter = meter.create_counter(
    "agent.tokens.input",
    description="Input/prompt tokens",
    unit="tokens"
)
output_token_counter = meter.create_counter(
    "agent.tokens.output", 
    description="Output/completion tokens",
    unit="tokens"
)

# Latency histogram - track request duration distribution
request_latency = meter.create_histogram(
    "agent.request.latency",
    description="Request latency in milliseconds",
    unit="ms"
)

# Tool calls counter - track which tools are used most
tool_calls_counter = meter.create_counter(
    "agent.tool.calls",
    description="Number of tool invocations",
    unit="calls"
)

# Error counter - track failures for alerting
error_counter = meter.create_counter(
    "agent.errors",
    description="Number of errors",
    unit="errors"
)

# =============================================================================
# TOKEN USAGE EXTRACTION
# =============================================================================
def record_token_usage(response, span):
    """
    Extract token usage from LangChain response and record as metrics + span attributes.
    
    Azure OpenAI returns token counts in response_metadata.
    We record both as metrics (for dashboards) and span attributes (for traces).
    """
    try:
        # LangChain stores Azure OpenAI metadata in response_metadata
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            
            # Token usage can be under 'token_usage' or 'usage' depending on version
            usage = metadata.get('token_usage', {}) or metadata.get('usage', {})
            
            # Extract individual token counts
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
            
            if total_tokens > 0:
                # Record as METRICS (for dashboards and alerting)
                # Labels/attributes allow filtering (e.g., by model)
                input_token_counter.add(input_tokens, {"model": "gpt-5.2"})
                output_token_counter.add(output_tokens, {"model": "gpt-5.2"})
                token_counter.add(total_tokens, {"model": "gpt-5.2"})
                
                # Record as SPAN ATTRIBUTES (visible in trace details)
                span.set_attribute("llm.tokens.input", input_tokens)
                span.set_attribute("llm.tokens.output", output_tokens)
                span.set_attribute("llm.tokens.total", total_tokens)
                
                # Also log for console visibility
                logger.info(f"Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                return total_tokens
                
    except Exception as e:
        # Don't fail the request if token extraction fails
        logger.warning(f"Could not extract token usage: {e}")
    return 0

# =============================================================================
# 3. LOGGING SETUP
# =============================================================================
# OTEL can also collect logs and correlate them with traces
# This sends Python logs to SigNoz alongside traces and metrics

# LoggerProvider manages log export
logger_provider = LoggerProvider(resource=resource)

# OTLPLogExporter sends logs to SigNoz
log_exporter = OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=True)

# BatchLogRecordProcessor batches logs before sending
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

# Register as global default
set_logger_provider(logger_provider)

# Create a LoggingHandler that bridges Python logging to OTEL
otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)

# Create our application logger
logger = logging.getLogger("weather-agent")
logger.setLevel(logging.INFO)

# Add OTEL handler - sends logs to SigNoz
logger.addHandler(otel_handler)

# Also add console handler - prints logs to terminal
logger.addHandler(logging.StreamHandler())

# =============================================================================
# SENSITIVE DATA REDACTION
# =============================================================================
# CRITICAL: Never log API keys, tokens, or secrets!
# These patterns detect and redact common sensitive data formats

SENSITIVE_PATTERNS = [
    # Google API keys start with 'AIza'
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***REDACTED***'),
    
    # OpenAI API keys start with 'sk-'
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***REDACTED***'),
    
    # Generic pattern for key=value pairs with sensitive names
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), 
     r'\1=***REDACTED***'),
]

def redact(text: str) -> str:
    """
    Remove sensitive data from text before logging or tracing.
    
    This is CRITICAL for security - API keys in logs can be stolen!
    """
    if not text or not isinstance(text, str):
        return text
    
    # Apply each pattern to redact sensitive data
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# =============================================================================
# WEATHER TOOLS
# =============================================================================
# Tools are functions the LLM can call to get real-world data
# The @tool decorator makes them available to LangChain agents

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    
    This is a mock implementation - in production, you'd call a real weather API.
    The LLM sees this docstring and uses it to understand when to call this tool.
    """
    # Mock weather data for demo purposes
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "new york": "18°C, Clear, Breezy",
        "london": "12°C, Overcast, Light Drizzle",
        "tokyo": "22°C, Sunny, Humid",
    }
    
    # Normalize city name for lookup
    city_lower = city.lower().strip()
    
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    
    # Return helpful error if city not found
    return f"Weather not available for {city}. Try: {', '.join(weather_data.keys())}"

@tool
def get_forecast(city: str, days: int = 3) -> str:
    """
    Get weather forecast for a city.
    
    Args:
        city: Name of the city
        days: Number of days to forecast (default 3, max 3)
    """
    # Mock forecast data
    forecasts = {
        "mumbai": ["32°C Sunny", "31°C Cloudy", "30°C Rain"],
        "delhi": ["28°C Clear", "30°C Sunny", "29°C Hazy"],
        "bangalore": ["24°C Rain", "23°C Cloudy", "25°C Pleasant"],
        "new york": ["17°C Sunny", "19°C Clear", "16°C Cloudy"],
        "london": ["11°C Rain", "10°C Overcast", "12°C Cloudy"],
    }
    
    city_lower = city.lower().strip()
    
    if city_lower in forecasts:
        # Limit to available days
        forecast_list = forecasts[city_lower][:min(days, 3)]
        # Format as readable string
        return f"Forecast for {city.title()}: " + ", ".join(
            [f"Day {i+1}: {f}" for i, f in enumerate(forecast_list)]
        )
    
    return f"Forecast not available for {city}."

# List of tools available to the LLM
tools = [get_weather, get_forecast]

# System prompt tells the LLM how to behave
SYSTEM_PROMPT = """You are a weather assistant. Use get_weather for current conditions and get_forecast for predictions.
Available cities: Mumbai, Delhi, Bangalore, New York, London, Tokyo. Be concise."""

# =============================================================================
# LLM SETUP
# =============================================================================
def get_llm():
    """
    Create and configure the Azure OpenAI LLM client.
    
    Returns None if API key is not set.
    Uses environment variables for configuration (security best practice).
    """
    # Get API key from environment (never hardcode!)
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not azure_api_key:
        return None
    
    # Create Azure OpenAI client with LangChain
    llm = AzureChatOpenAI(
        # Azure OpenAI endpoint URL
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sre-resources.openai.azure.com"),
        # Deployment name (model deployment in Azure)
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"),
        # API version
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        # API key for authentication
        api_key=azure_api_key,
    )
    
    # bind_tools() makes our tools available to the LLM
    # The LLM can now decide to call these tools based on user queries
    return llm.bind_tools(tools)

# =============================================================================
# CHAT FUNCTION WITH FULL OTEL INSTRUMENTATION
# =============================================================================
def chat(message, history):
    """
    Process a chat message with full OpenTelemetry instrumentation.
    
    This function demonstrates:
    1. Creating spans to track request flow
    2. Recording metrics for monitoring
    3. Logging with OTEL correlation
    4. Handling tool calls from the LLM
    
    Args:
        message: User's input message
        history: List of (user_msg, assistant_msg) tuples from conversation
    
    Returns:
        Assistant's response string
    """
    # Get LLM client
    llm = get_llm()
    if not llm:
        return "❌ AZURE_OPENAI_API_KEY not set"
    
    # -------------------------------------------------------------------------
    # START PARENT SPAN
    # -------------------------------------------------------------------------
    # This span encompasses the entire request
    # All child spans (LLM calls, tool calls) will be nested under it
    with tracer.start_as_current_span("agent_request") as span:
        # Record start time for latency calculation
        start_time = time.time()
        
        # Set span attributes - these appear in trace details in SigNoz
        span.set_attribute("user.input", message[:200])  # Truncate long messages
        span.set_attribute("llm.provider", "azure_openai")
        span.set_attribute("llm.model", "gpt-5.2")
        
        # Log the request (goes to both console and SigNoz)
        logger.info(f"Processing request: {message[:100]}...")
        
        # ---------------------------------------------------------------------
        # BUILD MESSAGE HISTORY
        # ---------------------------------------------------------------------
        # LangChain uses message objects to represent conversation
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Add conversation history
        # Gradio 4.x uses list of dicts: [{"role": "user", "content": "..."}, ...]
        for h in history:
            if isinstance(h, dict):
                # New Gradio format: dict with 'role' and 'content'
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=h.get("content", "")))
                elif h.get("role") == "assistant":
                    messages.append(AIMessage(content=h.get("content", "")))
            else:
                # Old Gradio format: tuple (user_msg, assistant_msg)
                messages.append(HumanMessage(content=h[0]))
                if h[1]:
                    messages.append(AIMessage(content=h[1]))
        
        # Add current user message
        messages.append(HumanMessage(content=message))
        
        try:
            # -----------------------------------------------------------------
            # FIRST LLM CALL
            # -----------------------------------------------------------------
            # Create a child span for the LLM call
            with tracer.start_as_current_span("llm_call") as llm_span:
                llm_start = time.time()
                llm_span.set_attribute("llm.model", "gpt-5.2")
                
                # Record the request (input to LLM)
                request_text = "\n".join([f"{m.type}: {m.content[:200]}" for m in messages])
                llm_span.set_attribute("llm.request", request_text[:1000])
                
                # Call the LLM
                response = llm.invoke(messages)
                
                # Record duration
                llm_duration = (time.time() - llm_start) * 1000
                llm_span.set_attribute("llm.duration_ms", llm_duration)
                
                # Record the response
                if response.tool_calls:
                    # LLM decided to call tools
                    llm_span.set_attribute("llm.response", f"Tool calls: {[tc['name'] for tc in response.tool_calls]}")
                else:
                    # LLM gave a direct response
                    llm_span.set_attribute("llm.response", response.content[:500])
                
                # Extract and record token usage
                record_token_usage(response, llm_span)
                
                # Mark span as successful
                llm_span.set_status(Status(StatusCode.OK))
                logger.info(f"LLM call completed in {llm_duration:.0f}ms")
            
            # -----------------------------------------------------------------
            # HANDLE TOOL CALLS
            # -----------------------------------------------------------------
            # If the LLM decided to call tools, execute them
            if response.tool_calls:
                # Add the assistant's response with tool calls ONCE before processing
                messages.append(response)
                
                # Process each tool call and collect results
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Create a span for each tool call
                    with tracer.start_as_current_span(f"tool_{tool_name}") as tool_span:
                        # Record tool details
                        tool_span.set_attribute("tool.name", tool_name)
                        tool_span.set_attribute("tool.args", str(tool_args))
                        
                        # Increment tool call counter metric
                        tool_calls_counter.add(1, {"tool": tool_name})
                        
                        logger.info(f"Calling tool: {tool_name}")
                        
                        # Execute the appropriate tool
                        if tool_name == "get_weather":
                            result = get_weather.invoke(tool_args)
                        elif tool_name == "get_forecast":
                            result = get_forecast.invoke(tool_args)
                        else:
                            result = "Unknown tool"
                        
                        # Record tool result
                        tool_span.set_attribute("tool.result", result[:200])
                        tool_span.set_status(Status(StatusCode.OK))
                        
                        # Add tool result message (one per tool call)
                        messages.append(ToolMessage(
                            content=result, 
                            tool_call_id=tool_call["id"]
                        ))
                
                # -------------------------------------------------------------
                # SECOND LLM CALL (after tools)
                # -------------------------------------------------------------
                # LLM needs to process tool results and generate final response
                with tracer.start_as_current_span("llm_call_final") as llm_span:
                    llm_span.set_attribute("llm.model", "gpt-5.2")
                    
                    # Record request with tool results
                    request_text = "\n".join([f"{m.type}: {str(m.content)[:200]}" for m in messages[-3:]])
                    llm_span.set_attribute("llm.request", request_text[:1000])
                    
                    # Call LLM again with tool results
                    response = llm.invoke(messages)
                    
                    # Record response
                    llm_span.set_attribute("llm.response", response.content[:500])
                    
                    # Record token usage for this call too
                    record_token_usage(response, llm_span)
                    llm_span.set_status(Status(StatusCode.OK))
            
            # -----------------------------------------------------------------
            # RECORD SUCCESS METRICS
            # -----------------------------------------------------------------
            duration_ms = (time.time() - start_time) * 1000
            
            # Set final span attributes
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.response", response.content[:200])
            span.set_status(Status(StatusCode.OK))
            
            # Record latency metric (histogram for percentile analysis)
            request_latency.record(duration_ms, {"status": "success"})
            
            logger.info(f"Request completed in {duration_ms:.0f}ms")
            
            return response.content
            
        except Exception as e:
            # -----------------------------------------------------------------
            # ERROR HANDLING
            # -----------------------------------------------------------------
            # IMPORTANT: Redact error message to avoid leaking API keys!
            safe_error = redact(str(e))
            
            # Mark span as failed
            span.set_status(Status(StatusCode.ERROR, safe_error))
            span.set_attribute("error.message", safe_error)
            
            # Increment error counter metric
            error_counter.add(1, {"type": type(e).__name__})
            
            logger.error(f"Error: {safe_error}")
            
            return f"❌ Error: {safe_error}"

# =============================================================================
# GRADIO UI
# =============================================================================
# Gradio provides a simple way to create web UIs for ML models

demo = gr.ChatInterface(
    fn=chat,                                    # Function to call for each message
    title="🌤️ Demo Weather Agent with O11Y",   # Page title
    description="Ask about weather. Traces, metrics, and logs are sent to SigNoz at http://localhost:8080",
    examples=[                                  # Example queries shown as buttons
        "What's the weather in Mumbai?", 
        "Forecast for New York"
    ],
)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Log startup information
    logger.info("Starting Weather Agent with OpenTelemetry")
    logger.info(f"OTLP Endpoint: {OTLP_ENDPOINT}")
    logger.info(f"Service Name: {SERVICE_NAME}")
    
    # Launch Gradio server on port 8000
    demo.launch(server_port=8000)
