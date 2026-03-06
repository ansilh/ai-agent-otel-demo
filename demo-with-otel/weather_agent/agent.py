"""
=============================================================================
DEMO: Weather Agent WITH OpenTelemetry Observability
=============================================================================

This is the same weather agent as demo-without-otel, but now instrumented
with OpenTelemetry to capture traces, metrics, and logs.

The agent is now a "GLASS BOX" - we can see:
✅ Token usage per request (input, output, total)
✅ Latency for each operation
✅ Tool calls with parameters and results
✅ Full request flow as traces
✅ Errors with context

Run with: adk web
Then open http://localhost:8000 to interact with the agent.
Check SigNoz at http://localhost:3301 to see the observability data.
=============================================================================
"""

import time
from functools import wraps

# ADK imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import tool

# Import our OTEL setup - this initializes tracing/metrics at import time
from .otel_setup import tracer, agent_metrics, logger, Status, StatusCode


# -----------------------------------------------------------------------------
# Instrumented Tool Decorator
# -----------------------------------------------------------------------------
# This decorator wraps tools to automatically create spans and record metrics

def instrumented_tool(func):
    """
    Decorator that adds OpenTelemetry instrumentation to a tool function.
    
    It automatically:
    - Creates a span for the tool call
    - Records tool name, inputs, and outputs as span attributes
    - Records tool call metrics
    - Logs the tool invocation
    - Handles errors with proper span status
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        
        # Create a span for this tool call
        with tracer.start_as_current_span(f"tool_{tool_name}") as span:
            # Record tool metadata
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.inputs", str(kwargs))
            
            # Log the tool call
            logger.info(f"Tool called: {tool_name} with {kwargs}")
            
            # Record metric
            agent_metrics["tool_calls"].add(1, {"tool": tool_name})
            
            try:
                # Execute the actual tool
                start_time = time.time()
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record result and timing
                span.set_attribute("tool.output", str(result)[:500])  # Truncate long outputs
                span.set_attribute("tool.duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))
                
                logger.info(f"Tool {tool_name} completed in {duration_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                # Record error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                agent_metrics["errors"].add(1, {"tool": tool_name, "error_type": type(e).__name__})
                logger.error(f"Tool {tool_name} failed: {e}")
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
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the current weather conditions
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
    
    Args:
        city: The name of the city
        days: Number of days to forecast (default: 3)
        
    Returns:
        A string with the weather forecast
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
        return f"Forecast not available for {city}"


# -----------------------------------------------------------------------------
# Custom LLM Wrapper with Token Tracking
# -----------------------------------------------------------------------------
# ADK's LiteLlm doesn't expose token counts directly, so we create a wrapper
# that intercepts calls and records token usage from the response metadata.

class InstrumentedLiteLlm(LiteLlm):
    """
    A LiteLlm wrapper that adds OpenTelemetry instrumentation.
    
    This captures:
    - Token usage (input, output, total) per LLM call
    - Latency for each LLM call
    - Model information
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def generate(self, *args, **kwargs):
        """
        Wrap the generate method to add tracing.
        """
        with tracer.start_as_current_span("llm_generate") as span:
            span.set_attribute("llm.model", self.model_string)
            
            start_time = time.time()
            
            try:
                # Call the parent generate method
                response = await super().generate(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                
                # Try to extract token usage if available
                # Note: Token extraction depends on the response structure
                # which varies by model/provider
                if hasattr(response, 'usage'):
                    input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                    
                    span.set_attribute("llm.tokens.input", input_tokens)
                    span.set_attribute("llm.tokens.output", output_tokens)
                    span.set_attribute("llm.tokens.total", input_tokens + output_tokens)
                    
                    # Record metrics
                    agent_metrics["input_tokens"].add(input_tokens, {"model": self.model_string})
                    agent_metrics["output_tokens"].add(output_tokens, {"model": self.model_string})
                    
                    logger.info(f"LLM call: input={input_tokens}, output={output_tokens}, latency={duration_ms:.2f}ms")
                
                span.set_status(Status(StatusCode.OK))
                return response
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                agent_metrics["errors"].add(1, {"component": "llm", "error_type": type(e).__name__})
                raise


# -----------------------------------------------------------------------------
# Create the Instrumented Agent
# -----------------------------------------------------------------------------

# Use the instrumented LLM wrapper
instrumented_model = InstrumentedLiteLlm(model_string="gemini/gemini-1.5-flash")

root_agent = Agent(
    name="weather_agent_otel",
    model=instrumented_model,
    instruction="""You are a helpful weather assistant with full observability. You can:

1. Get current weather for cities in India using the get_weather tool
2. Get weather forecasts using the get_forecast tool

When users ask about weather:
- Use get_weather for current conditions
- Use get_forecast for future predictions
- Be friendly and provide helpful context about the weather

If asked about cities not in your database, let the user know which cities are available.

Always be concise but informative in your responses.

Note: All your operations are being traced for observability demonstration.""",
    
    tools=[get_weather, get_forecast],
)


# =============================================================================
# WHAT'S NOW VISIBLE IN SIGNOZ
# =============================================================================
#
# After running this agent, check SigNoz at http://localhost:3301:
#
# TRACES:
# - Parent span: agent_request (entire conversation turn)
# - Child spans: llm_generate (each LLM call)
# - Child spans: tool_get_weather, tool_get_forecast (tool calls)
# - Attributes: model, tokens, latency, inputs, outputs
#
# METRICS:
# - agent.tokens.input: Total input tokens (counter)
# - agent.tokens.output: Total output tokens (counter)
# - agent.request.latency: Request latency distribution (histogram)
# - agent.tool.calls: Tool call counts by tool name (counter)
# - agent.errors: Error counts by type (counter)
#
# LOGS:
# - Tool invocations with parameters
# - LLM call completions with token counts
# - Errors with context
#
# This is the "GLASS BOX" - full visibility into agent behavior!
# =============================================================================
