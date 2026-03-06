"""
=============================================================================
DEMO: Simple Weather Agent WITHOUT OpenTelemetry
=============================================================================

This is a basic ADK agent with NO observability instrumentation.
It demonstrates the "black box" problem - the agent works, but we have
no visibility into what's happening inside.

Run with: adk web
Then open http://localhost:8000 to interact with the agent.
=============================================================================
"""

# Import the ADK Agent class - this is the base class for all ADK agents
from google.adk.agents import Agent

# Import Gemini model for LLM capabilities
from google.adk.models.lite_llm import LiteLlm

# Import FunctionTool to wrap our functions as tools
from google.adk.tools import FunctionTool


# -----------------------------------------------------------------------------
# Define Tools
# -----------------------------------------------------------------------------
# Tools are functions that the agent can decide to call based on user queries.
# The agent doesn't know real-time data, but it can call tools to get it.
# In ADK, we define plain functions and wrap them with FunctionTool.

def get_weather(city: str) -> str:
    """
    Get the current weather for a specified city.
    
    This is a mock implementation. In production, you would call
    a real weather API like OpenWeatherMap or WeatherAPI.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the current weather conditions
    """
    # Mock weather data - replace with real API call in production
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "chennai": "34°C, Hot and Humid",
        "kolkata": "30°C, Humid, Overcast",
        "hyderabad": "29°C, Warm, Clear Skies",
        "pune": "26°C, Pleasant, Partly Cloudy",
    }
    
    # Normalize city name and look up weather
    city_lower = city.lower().strip()
    
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}. Available cities: {', '.join(weather_data.keys())}"


def get_forecast(city: str, days: int = 3) -> str:
    """
    Get weather forecast for the next few days.
    
    Args:
        city: The name of the city
        days: Number of days to forecast (default: 3)
        
    Returns:
        A string with the weather forecast
    """
    # Mock forecast data
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
# Create the Agent
# -----------------------------------------------------------------------------
# The root_agent is the main entry point that ADK web will use.
# It must be named 'root_agent' for ADK to discover it.

root_agent = Agent(
    # A unique name for this agent
    name="weather_agent",
    
    # The LLM model to use for reasoning
    # Using Gemini 1.5 Flash for fast responses
    model=LiteLlm(model_string="gemini/gemini-1.5-flash"),
    
    # System instruction that defines the agent's behavior
    # This is the "personality" and capabilities of the agent
    instruction="""You are a helpful weather assistant. You can:

1. Get current weather for cities in India using the get_weather tool
2. Get weather forecasts using the get_forecast tool

When users ask about weather:
- Use get_weather for current conditions
- Use get_forecast for future predictions
- Be friendly and provide helpful context about the weather

If asked about cities not in your database, let the user know which cities are available.

Always be concise but informative in your responses.""",
    
    # Register the tools this agent can use
    # Wrap functions with FunctionTool for ADK
    tools=[FunctionTool(func=get_weather), FunctionTool(func=get_forecast)],
)


# =============================================================================
# THE PROBLEM: NO OBSERVABILITY
# =============================================================================
# 
# This agent works fine, but we have NO visibility into:
#
# ❌ How many tokens were used for each request?
# ❌ How long did the LLM take to respond?
# ❌ Which tools were called and with what parameters?
# ❌ If something fails, where exactly did it fail?
# ❌ What was the full conversation flow?
#
# This is the "BLACK BOX" problem.
#
# In the next demo (demo-with-otel), we'll add OpenTelemetry
# to turn this into a "GLASS BOX" where we can see everything.
# =============================================================================
