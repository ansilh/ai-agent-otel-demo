"""
=============================================================================
DEMO: Weather Agent using Chainlit WITH OpenTelemetry Observability
=============================================================================

This is the same weather agent as chainlit-without-otel, but now instrumented
with OpenTelemetry to capture traces, metrics, and logs.

The agent is now a "GLASS BOX" - we can see:
✅ Token usage per request (input, output, total)
✅ Latency for each operation
✅ Tool calls with parameters and results
✅ Full request flow as traces
✅ Errors with context

Run with: chainlit run app.py
Then open http://localhost:8000 to interact with the agent.
Check SigNoz at http://localhost:3301 to see the observability data.
=============================================================================
"""

import os
import time
from functools import wraps

# Chainlit provides the web chat interface
import chainlit as cl

# LangChain components for building the agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler

# -----------------------------------------------------------------------------
# OpenTelemetry Setup
# -----------------------------------------------------------------------------
# Import and configure OTEL before anything else

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
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "chainlit-weather-agent")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

# Create resource that identifies this service
resource = Resource.create({
    "service.name": SERVICE_NAME,
    "service.version": "1.0.0",
    "deployment.environment": "demo",
})

# Set up Tracing
tracer_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("chainlit-weather-agent", "1.0.0")

# Set up Metrics
metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)
metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("chainlit-weather-agent", "1.0.0")

# Create metric instruments
token_counter = meter.create_counter(
    name="agent.tokens.total",
    description="Total tokens used",
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
    """Decorator that adds OTEL tracing to tool functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        
        with tracer.start_as_current_span(f"tool_{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.inputs", str(kwargs))
            
            tool_calls_counter.add(1, {"tool": tool_name})
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                span.set_attribute("tool.output", str(result)[:500])
                span.set_attribute("tool.duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))
                
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
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
    }
    
    city_lower = city.lower().strip()
    
    if city_lower in forecasts:
        forecast_list = forecasts[city_lower][:min(days, 5)]
        forecast_str = "\n".join([f"  Day {i+1}: {f}" for i, f in enumerate(forecast_list)])
        return f"Forecast for {city.title()} (next {len(forecast_list)} days):\n{forecast_str}"
    else:
        return f"Forecast not available for {city}. Available cities: mumbai, delhi, bangalore"


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
        self.current_span.set_attribute("llm.model", "qwen2.5:7b")
        self.current_span.set_attribute("llm.prompt_length", sum(len(p) for p in prompts))
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes generating."""
        if self.current_span:
            duration_ms = (time.time() - self.llm_start_time) * 1000
            self.current_span.set_attribute("llm.duration_ms", duration_ms)
            
            # Try to get token usage from response
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                
                self.current_span.set_attribute("llm.tokens.input", input_tokens)
                self.current_span.set_attribute("llm.tokens.output", output_tokens)
                self.current_span.set_attribute("llm.tokens.total", input_tokens + output_tokens)
                
                token_counter.add(input_tokens + output_tokens, {"model": "qwen2.5:7b"})
            
            self.current_span.set_status(Status(StatusCode.OK))
            self.current_span.end()
            self.current_span = None
    
    def on_llm_error(self, error, **kwargs):
        """Called when LLM encounters an error."""
        if self.current_span:
            self.current_span.set_status(Status(StatusCode.ERROR, str(error)))
            self.current_span.record_exception(error)
            self.current_span.end()
            self.current_span = None


# -----------------------------------------------------------------------------
# Create the Agent
# -----------------------------------------------------------------------------

tools = [get_weather, get_forecast]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful weather assistant with full observability. You can:

1. Get current weather for cities in India using the get_weather tool
2. Get weather forecasts using the get_forecast tool

When users ask about weather:
- Use get_weather for current conditions
- Use get_forecast for future predictions
- Be friendly and provide helpful context about the weather

If asked about cities not in your database, let the user know which cities are available.
Always be concise but informative in your responses.

Note: All your operations are being traced for observability demonstration."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# -----------------------------------------------------------------------------
# Chainlit Event Handlers
# -----------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    """Initialize the LLM and agent with OTEL instrumentation."""
    
    # Create the LLM with OTEL callback
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        callbacks=[OTELCallbackHandler()],
    )
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[OTELCallbackHandler()],
    )
    
    # Store in session
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_history", [])
    
    # Welcome message
    await cl.Message(
        content="👋 Hello! I'm your weather assistant with **full observability**. Ask me about weather in Indian cities!\n\n📊 Check SigNoz at http://localhost:3301 to see traces, metrics, and logs."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages with OTEL tracing."""
    
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history")
    
    # Create a parent span for the entire request
    with tracer.start_as_current_span("agent_request") as span:
        span.set_attribute("user.input", message.content)
        
        start_time = time.time()
        
        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()
        
        try:
            # Run the agent
            response = await agent_executor.ainvoke({
                "input": message.content,
                "chat_history": chat_history,
            })
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics and span attributes
            span.set_attribute("agent.response", response["output"][:500])
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_status(Status(StatusCode.OK))
            
            request_latency.record(duration_ms, {"status": "success"})
            
            # Update chat history
            chat_history.append(("human", message.content))
            chat_history.append(("assistant", response["output"]))
            cl.user_session.set("chat_history", chat_history)
            
            # Send the response
            msg.content = response["output"]
            await msg.update()
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            request_latency.record((time.time() - start_time) * 1000, {"status": "error"})
            
            msg.content = f"❌ Error: {str(e)}"
            await msg.update()


# =============================================================================
# WHAT'S NOW VISIBLE IN SIGNOZ
# =============================================================================
#
# After running this agent, check SigNoz at http://localhost:3301:
#
# TRACES:
# - Parent span: agent_request (entire conversation turn)
# - Child spans: llm_call (each LLM call)
# - Child spans: tool_get_weather, tool_get_forecast (tool calls)
# - Attributes: model, tokens, latency, inputs, outputs
#
# METRICS:
# - agent.tokens.total: Total tokens used (counter)
# - agent.request.latency: Request latency distribution (histogram)
# - agent.tool.calls: Tool call counts by tool name (counter)
#
# This is the "GLASS BOX" - full visibility into agent behavior!
# =============================================================================
