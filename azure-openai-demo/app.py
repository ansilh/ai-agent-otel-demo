"""
=============================================================================
DEMO: Weather Agent using Azure OpenAI with OpenTelemetry
=============================================================================

This demo uses:
- LangChain (vendor-neutral, open-source) for agent framework
- Chainlit (open-source) for web UI
- OpenTelemetry (open-source, vendor-neutral) for observability
- Azure OpenAI as the LLM provider

The agent is a "GLASS BOX" with full observability:
✅ Token usage per request
✅ Latency for each operation
✅ Tool calls with parameters and results
✅ Full request flow as traces

Run with: chainlit run app.py
Then open http://localhost:8000 to interact with the agent.
Check SigNoz at http://localhost:3301 to see the observability data.
=============================================================================
"""

import os
import sys
import time
import re
from functools import wraps

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Chainlit - Open source web UI framework
import chainlit as cl

# -----------------------------------------------------------------------------
# Sensitive Data Redaction (CRITICAL FOR SECURITY)
# -----------------------------------------------------------------------------
# NEVER log API keys, tokens, or credentials in traces, metrics, or logs

SENSITIVE_PATTERNS = [
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***GOOGLE_API_KEY***'),
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***OPENAI_KEY***'),
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password|credential|auth)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), r'\1=***REDACTED***'),
    (re.compile(r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]+)'), r'\1***REDACTED***'),
    (re.compile(r'(https?://)[^:]+:[^@]+@'), r'\1***:***@'),
    (re.compile(r'(?i)(\?|&)(api[_-]?key|key|token)=([^&\s]+)'), r'\1\2=***REDACTED***'),
]

def redact_sensitive(text: str) -> str:
    """Redact sensitive data from text. ALWAYS use this before logging."""
    if not text or not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result

def safe_str(obj, max_length: int = 500) -> str:
    """Convert to string with redaction and length limit."""
    try:
        text = str(obj)
        redacted = redact_sensitive(text)
        return redacted[:max_length] + '...' if len(redacted) > max_length else redacted
    except Exception:
        return '[UNABLE TO CONVERT]'

# LangChain - Open source, vendor-neutral agent framework
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler

# -----------------------------------------------------------------------------
# OpenTelemetry Setup (Open Source, Vendor Neutral)
# -----------------------------------------------------------------------------
# OTEL is the industry standard for observability - works with any backend

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.trace import Status, StatusCode

# Configuration from environment variables
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "azure-openai-weather-agent")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

# Create resource that identifies this service in observability backend
resource = Resource.create({
    "service.name": SERVICE_NAME,
    "service.version": "1.0.0",
    "deployment.environment": "demo",
    "cloud.provider": "azure",
})

# Set up Tracing - traces show request flow through the system
tracer_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("azure-openai-weather-agent", "1.0.0")

# Set up Metrics - numerical measurements over time
metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)
metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("azure-openai-weather-agent", "1.0.0")

# Create metric instruments
token_counter = meter.create_counter(
    name="agent.tokens.total",
    description="Total tokens used",
    unit="tokens",
)
input_token_counter = meter.create_counter(
    name="agent.tokens.input",
    description="Input/prompt tokens used",
    unit="tokens",
)
output_token_counter = meter.create_counter(
    name="agent.tokens.output",
    description="Output/completion tokens used",
    unit="tokens",
)
request_latency = meter.create_histogram(
    name="agent.request.latency",
    description="Request latency in milliseconds",
    unit="ms",
)
tool_calls_counter = meter.create_counter(
    name="agent.tool.calls",
    description="Number of tool calls",
    unit="calls",
)


# -----------------------------------------------------------------------------
# Instrumented Tool Decorator
# -----------------------------------------------------------------------------

def instrumented_tool(func):
    """Decorator that adds OTEL tracing to tool functions with safe redaction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        
        # Create a span for this tool call
        with tracer.start_as_current_span(f"tool_{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            # SECURITY: Redact sensitive data from inputs before logging
            span.set_attribute("tool.inputs", safe_str(kwargs))
            
            # Record metric
            tool_calls_counter.add(1, {"tool": tool_name})
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record result in span (SECURITY: redact sensitive data)
                span.set_attribute("tool.output", safe_str(result))
                span.set_attribute("tool.duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))
                
                return result
            except Exception as e:
                # SECURITY: Redact error messages before logging
                span.set_status(Status(StatusCode.ERROR, redact_sensitive(str(e))))
                span.record_exception(e)
                raise
    
    return wrapper


# -----------------------------------------------------------------------------
# Define Instrumented Tools
# -----------------------------------------------------------------------------

@tool
@instrumented_tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a specified city.
    Use this when the user asks about current weather conditions.
    
    Args:
        city: The name of the city to get weather for
    """
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "chennai": "34°C, Hot and Humid",
        "kolkata": "30°C, Humid, Overcast",
        "hyderabad": "29°C, Warm, Clear Skies",
        "pune": "26°C, Pleasant, Partly Cloudy",
        "new york": "18°C, Clear, Breezy",
        "london": "12°C, Overcast, Light Drizzle",
        "tokyo": "22°C, Sunny, Humid",
    }
    
    city_lower = city.lower().strip()
    
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}. Available cities: {', '.join(weather_data.keys())}"


@tool
@instrumented_tool
def get_forecast(city: str, days: int = 3) -> str:
    """
    Get weather forecast for the next few days.
    Use this when the user asks about future weather predictions.
    
    Args:
        city: The name of the city
        days: Number of days to forecast (default: 3, max: 5)
    """
    forecasts = {
        "mumbai": ["32°C Sunny", "31°C Cloudy", "30°C Rain", "29°C Thunderstorm", "31°C Partly Cloudy"],
        "delhi": ["28°C Clear", "30°C Sunny", "29°C Hazy", "27°C Cloudy", "28°C Clear"],
        "bangalore": ["24°C Rain", "23°C Cloudy", "25°C Pleasant", "24°C Light Rain", "26°C Sunny"],
        "new york": ["17°C Sunny", "19°C Clear", "16°C Cloudy", "15°C Rain", "18°C Partly Cloudy"],
        "london": ["11°C Rain", "10°C Overcast", "12°C Cloudy", "13°C Partly Cloudy", "11°C Drizzle"],
    }
    
    city_lower = city.lower().strip()
    
    if city_lower in forecasts:
        forecast_list = forecasts[city_lower][:min(days, 5)]
        forecast_str = "\n".join([f"  Day {i+1}: {f}" for i, f in enumerate(forecast_list)])
        return f"Forecast for {city.title()} (next {len(forecast_list)} days):\n{forecast_str}"
    else:
        return f"Forecast not available for {city}. Available cities: mumbai, delhi, bangalore, new york, london"


# -----------------------------------------------------------------------------
# LangChain Callback Handler for OTEL
# -----------------------------------------------------------------------------

class OTELCallbackHandler(BaseCallbackHandler):
    """Callback handler that creates OTEL spans for LLM calls."""
    
    def __init__(self):
        self.llm_start_time = None
        self.current_span = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts generating."""
        self.llm_start_time = time.time()
        self.current_span = tracer.start_span("llm_call")
        self.current_span.set_attribute("llm.provider", "azure_openai")
        self.current_span.set_attribute("llm.model", "gpt-5.2")
        self.current_span.set_attribute("llm.prompt_length", sum(len(p) for p in prompts))
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes generating."""
        if self.current_span:
            duration_ms = (time.time() - self.llm_start_time) * 1000
            self.current_span.set_attribute("llm.duration_ms", duration_ms)
            
            # Extract token usage from Azure OpenAI response
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                total_tokens = token_usage.get('total_tokens', input_tokens + output_tokens)
                
                # Record in span
                self.current_span.set_attribute("llm.tokens.input", input_tokens)
                self.current_span.set_attribute("llm.tokens.output", output_tokens)
                self.current_span.set_attribute("llm.tokens.total", total_tokens)
                
                # Record metrics
                input_token_counter.add(input_tokens, {"model": "gpt-5.2", "provider": "azure"})
                output_token_counter.add(output_tokens, {"model": "gpt-5.2", "provider": "azure"})
                token_counter.add(total_tokens, {"model": "gpt-5.2", "provider": "azure"})
            
            self.current_span.set_status(Status(StatusCode.OK))
            self.current_span.end()
            self.current_span = None
    
    def on_llm_error(self, error, **kwargs):
        """Called when LLM encounters an error."""
        if self.current_span:
            # SECURITY: Redact error messages - they may contain API keys in URLs
            self.current_span.set_status(Status(StatusCode.ERROR, redact_sensitive(str(error))))
            # Don't record full exception - it may contain sensitive data
            self.current_span.set_attribute("error.type", type(error).__name__)
            self.current_span.end()
            self.current_span = None


# -----------------------------------------------------------------------------
# Create the Agent
# -----------------------------------------------------------------------------

tools = [get_weather, get_forecast]

SYSTEM_PROMPT = """You are a helpful weather assistant powered by Azure OpenAI with full observability. You can:

1. Get current weather for cities using the get_weather tool
2. Get weather forecasts using the get_forecast tool

When users ask about weather:
- Use get_weather for current conditions
- Use get_forecast for future predictions
- Be friendly and provide helpful context about the weather

Available cities: Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune, New York, London, Tokyo

Always be concise but informative in your responses.

Note: All operations are being traced with OpenTelemetry for observability demonstration."""


# -----------------------------------------------------------------------------
# Chainlit Event Handlers
# -----------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    """Initialize the Azure OpenAI LLM and agent with OTEL instrumentation."""
    
    # Get Azure OpenAI configuration from environment variables
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sre-resources.openai.azure.com")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    
    if not azure_api_key:
        await cl.Message(
            content="❌ Error: AZURE_OPENAI_API_KEY environment variable not set.\n\nPlease set it and restart the application."
        ).send()
        return
    
    # Create the Azure OpenAI LLM using LangChain (vendor-neutral SDK)
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        api_key=azure_api_key,
        callbacks=[OTELCallbackHandler()],
    )
    
    # Create the agent using LangGraph (open-source)
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    
    # Store in session
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_history", [])
    
    # Welcome message
    await cl.Message(
        content="""👋 Hello! I'm your weather assistant powered by **Azure OpenAI (GPT-5.2)** with full observability.

Ask me about weather in cities like Mumbai, Delhi, New York, London, or Tokyo!

📊 **Observability**: Check SigNoz at http://localhost:3301 to see traces, metrics, and logs.

🔧 **Tech Stack**:
- LangChain (vendor-neutral agent framework)
- Chainlit (open-source web UI)
- OpenTelemetry (vendor-neutral observability)
- Azure OpenAI (LLM provider)"""
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages with OTEL tracing."""
    
    agent_executor = cl.user_session.get("agent_executor")
    
    if not agent_executor:
        await cl.Message(content="❌ Agent not initialized. Please check your Azure OpenAI configuration.").send()
        return
    
    chat_history = cl.user_session.get("chat_history")
    
    # Create a parent span for the entire request
    with tracer.start_as_current_span("agent_request") as span:
        # SECURITY: User input is generally safe, but redact just in case
        span.set_attribute("user.input", safe_str(message.content, max_length=1000))
        span.set_attribute("llm.provider", "azure_openai")
        span.set_attribute("llm.model", "gpt-5.2")
        
        start_time = time.time()
        
        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()
        
        try:
            # Run the agent (LangGraph uses "messages" format)
            from langchain_core.messages import HumanMessage
            
            # Build messages from chat history
            messages = []
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=message.content))
            
            response = await agent_executor.ainvoke({"messages": messages})
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract the final response from LangGraph output
            output_messages = response.get("messages", [])
            final_response = output_messages[-1].content if output_messages else "No response"
            
            # Record metrics and span attributes (SECURITY: redact response)
            span.set_attribute("agent.response", safe_str(final_response))
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_status(Status(StatusCode.OK))
            
            request_latency.record(duration_ms, {"status": "success", "provider": "azure"})
            
            # Update chat history
            chat_history.append(("human", message.content))
            chat_history.append(("assistant", final_response))
            cl.user_session.set("chat_history", chat_history)
            
            # Send the response
            msg.content = final_response
            await msg.update()
            
        except Exception as e:
            # SECURITY: CRITICAL - Error messages often contain API keys in URLs
            safe_error = redact_sensitive(str(e))
            span.set_status(Status(StatusCode.ERROR, safe_error))
            # Don't use record_exception - it captures full stack which may have secrets
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", safe_error)
            request_latency.record((time.time() - start_time) * 1000, {"status": "error", "provider": "azure"})
            
            # SECURITY: Redact error shown to user as well
            msg.content = f"❌ Error: {safe_error}"
            await msg.update()


# =============================================================================
# OBSERVABILITY DATA IN SIGNOZ
# =============================================================================
#
# After running this agent, check SigNoz at http://localhost:3301:
#
# TRACES:
# - Parent span: agent_request (entire conversation turn)
# - Child spans: llm_call (each Azure OpenAI call)
# - Child spans: tool_get_weather, tool_get_forecast (tool calls)
# - Attributes: provider=azure_openai, model=gpt-5.2, tokens, latency
#
# METRICS:
# - agent.tokens.total: Total tokens used
# - agent.tokens.input: Input/prompt tokens
# - agent.tokens.output: Output/completion tokens
# - agent.request.latency: Request latency distribution
# - agent.tool.calls: Tool call counts by tool name
#
# All using vendor-neutral, open-source SDKs!
# =============================================================================
