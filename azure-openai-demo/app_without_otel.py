"""
=============================================================================
DEMO: Weather Agent WITHOUT OpenTelemetry (O11Y)
=============================================================================

This is the "black box" version - NO observability instrumentation.
Compare this with app_with_otel.py to see what observability adds.

Without O11Y, you cannot see:
- How long LLM calls take
- Token usage and costs
- Which tools are being called
- Error patterns and root causes
- Request flow through the system

Run with: python3 app_without_otel.py
UI: http://localhost:8000
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os      # For reading environment variables (API keys, endpoints)
import re      # For regex-based sensitive data redaction

# Gradio - Web UI framework for creating chat interfaces
# Provides a simple way to create web UIs for ML/AI applications
import gradio as gr

# LangChain - Framework for building LLM applications
# Provides abstractions for working with various LLM providers
from langchain_openai import AzureChatOpenAI  # Azure OpenAI client wrapper
from langchain_core.tools import tool          # Decorator to create callable tools
from langchain_core.messages import (          # Message types for chat history
    HumanMessage,    # Represents user's message in conversation
    AIMessage,       # Represents assistant's response
    SystemMessage,   # System prompt that sets assistant behavior
    ToolMessage,     # Result from tool execution
)

# =============================================================================
# SENSITIVE DATA REDACTION
# =============================================================================
# Even without O11Y, we should never expose API keys in error messages!
# This is a security best practice regardless of observability.

# Regex patterns to detect common sensitive data formats
SENSITIVE_PATTERNS = [
    # Google API keys start with 'AIza' followed by 35 characters
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***REDACTED***'),
    
    # OpenAI API keys start with 'sk-' followed by 48+ characters
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***REDACTED***'),
    
    # Generic pattern: key=value pairs where key contains sensitive words
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), 
     r'\1=***REDACTED***'),
]

def redact(text: str) -> str:
    """
    Remove sensitive data from text before displaying to users.
    
    This prevents accidental exposure of API keys in error messages.
    Error messages from API calls often contain the full URL including keys!
    
    Args:
        text: String that may contain sensitive data
        
    Returns:
        String with sensitive data replaced by '***REDACTED***'
    """
    # Handle None or non-string inputs
    if not text or not isinstance(text, str):
        return text
    
    # Apply each regex pattern to find and replace sensitive data
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    
    return text

# =============================================================================
# WEATHER TOOLS
# =============================================================================
# Tools are functions that the LLM can call to get real-world data.
# The @tool decorator from LangChain:
# 1. Registers the function as a callable tool
# 2. Uses the docstring to tell the LLM when to use it
# 3. Extracts parameter types from type hints

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    
    This is a mock implementation for demo purposes.
    In production, you would call a real weather API like OpenWeatherMap.
    
    The docstring is important - the LLM reads it to understand:
    - What this tool does
    - When to use it
    - What parameters it needs
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        String describing current weather conditions
    """
    # Mock weather data - in production, call a real API
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "new york": "18°C, Clear, Breezy",
        "london": "12°C, Overcast, Light Drizzle",
        "tokyo": "22°C, Sunny, Humid",
    }
    
    # Normalize city name (lowercase, strip whitespace) for lookup
    city_lower = city.lower().strip()
    
    # Return weather if city is found
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    
    # Return helpful error message listing available cities
    return f"Weather not available for {city}. Try: {', '.join(weather_data.keys())}"

@tool
def get_forecast(city: str, days: int = 3) -> str:
    """
    Get weather forecast for a city.
    
    Args:
        city: Name of the city to get forecast for
        days: Number of days to forecast (default 3, max 3 in this demo)
        
    Returns:
        String with multi-day forecast
    """
    # Mock forecast data
    forecasts = {
        "mumbai": ["32°C Sunny", "31°C Cloudy", "30°C Rain"],
        "delhi": ["28°C Clear", "30°C Sunny", "29°C Hazy"],
        "bangalore": ["24°C Rain", "23°C Cloudy", "25°C Pleasant"],
        "new york": ["17°C Sunny", "19°C Clear", "16°C Cloudy"],
        "london": ["11°C Rain", "10°C Overcast", "12°C Cloudy"],
    }
    
    # Normalize city name
    city_lower = city.lower().strip()
    
    if city_lower in forecasts:
        # Limit to available forecast days (max 3 in our mock data)
        forecast_list = forecasts[city_lower][:min(days, 3)]
        
        # Format as readable string with day numbers
        return f"Forecast for {city.title()}: " + ", ".join(
            [f"Day {i+1}: {f}" for i, f in enumerate(forecast_list)]
        )
    
    return f"Forecast not available for {city}."

# List of tools available to the LLM
# The LLM will see these tools and can decide to call them
tools = [get_weather, get_forecast]

# System prompt defines the assistant's behavior and capabilities
# This is sent with every request to set context
SYSTEM_PROMPT = """You are a weather assistant. Use get_weather for current conditions and get_forecast for predictions.
Available cities: Mumbai, Delhi, Bangalore, New York, London, Tokyo. Be concise."""

# =============================================================================
# LLM SETUP
# =============================================================================
def get_llm():
    """
    Create and configure the Azure OpenAI LLM client.
    
    Uses environment variables for configuration - this is a security best practice.
    Never hardcode API keys in source code!
    
    Environment variables needed:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key (required)
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL (optional, has default)
    - AZURE_OPENAI_DEPLOYMENT: Model deployment name (optional, has default)
    - AZURE_OPENAI_API_VERSION: API version (optional, has default)
    
    Returns:
        LLM client with tools bound, or None if API key not set
    """
    # Get API key from environment variable
    # Returns None if not set, which we handle gracefully
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    
    if not azure_api_key:
        return None
    
    # Create Azure OpenAI client using LangChain's wrapper
    # This provides a consistent interface regardless of the underlying provider
    llm = AzureChatOpenAI(
        # Azure OpenAI resource endpoint
        azure_endpoint=os.environ.get(
            "AZURE_OPENAI_ENDPOINT", 
            "https://sre-resources.openai.azure.com"
        ),
        # Deployment name - this is the model deployment in Azure
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"),
        # API version - Azure OpenAI uses dated versions
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        # API key for authentication
        api_key=azure_api_key,
    )
    
    # bind_tools() makes our tools available to the LLM
    # After this, the LLM can decide to call get_weather or get_forecast
    # based on the user's query
    return llm.bind_tools(tools)

# =============================================================================
# CHAT FUNCTION (NO OBSERVABILITY)
# =============================================================================
def chat(message, history):
    """
    Process a chat message and return a response.
    
    This is the "black box" version - no observability instrumentation.
    We have no visibility into:
    - How long each step takes
    - Token usage (cost tracking impossible)
    - Which tools are called and how often
    - Error patterns over time
    
    Args:
        message: User's input message (string)
        history: Conversation history as list of (user_msg, assistant_msg) tuples
        
    Returns:
        Assistant's response string
    """
    # Get LLM client
    llm = get_llm()
    
    # Handle missing API key gracefully
    if not llm:
        return "❌ AZURE_OPENAI_API_KEY not set. Please set the environment variable."
    
    # -------------------------------------------------------------------------
    # BUILD MESSAGE HISTORY
    # -------------------------------------------------------------------------
    # LangChain uses typed message objects to represent conversation
    # Start with system prompt to set assistant behavior
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add conversation history from previous turns
    # history is a list of (user_message, assistant_message) tuples
    for h in history:
        messages.append(HumanMessage(content=h[0]))  # User's message
        if h[1]:  # Assistant's response (may be None for current turn)
            messages.append(AIMessage(content=h[1]))
    
    # Add the current user message
    messages.append(HumanMessage(content=message))
    
    try:
        # ---------------------------------------------------------------------
        # FIRST LLM CALL
        # ---------------------------------------------------------------------
        # Send messages to LLM and get response
        # The LLM may either:
        # 1. Return a direct text response
        # 2. Request to call one or more tools
        response = llm.invoke(messages)
        
        # ---------------------------------------------------------------------
        # HANDLE TOOL CALLS
        # ---------------------------------------------------------------------
        # Check if LLM wants to call tools
        if response.tool_calls:
            # Process each tool call
            for tool_call in response.tool_calls:
                # Extract tool name and arguments
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Execute the appropriate tool
                if tool_name == "get_weather":
                    result = get_weather.invoke(tool_args)
                elif tool_name == "get_forecast":
                    result = get_forecast.invoke(tool_args)
                else:
                    result = "Unknown tool"
                
                # Add tool call and result to message history
                # This lets the LLM see what the tool returned
                messages.append(response)  # LLM's tool call request
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]  # Links result to request
                ))
            
            # -----------------------------------------------------------------
            # SECOND LLM CALL (after tools)
            # -----------------------------------------------------------------
            # Now that we have tool results, ask LLM to generate final response
            response = llm.invoke(messages)
        
        # Return the final response content
        return response.content
        
    except Exception as e:
        # ---------------------------------------------------------------------
        # ERROR HANDLING
        # ---------------------------------------------------------------------
        # IMPORTANT: Redact error message to avoid leaking API keys!
        # Error messages often contain full URLs with embedded credentials
        safe_error = redact(str(e))
        
        return f"❌ Error: {safe_error}"

# =============================================================================
# GRADIO UI
# =============================================================================
# Gradio provides a simple way to create web UIs for ML/AI applications
# ChatInterface specifically creates a chat-style interface

demo = gr.ChatInterface(
    # Function to call for each user message
    fn=chat,
    
    # Page title shown in browser tab and header
    title="🌤️ Demo Weather Agent (No O11Y)",
    
    # Description shown below title
    description="Ask about weather. This version has NO observability - it's a black box!",
    
    # Example queries shown as clickable buttons
    examples=[
        "What's the weather in Mumbai?", 
        "Forecast for New York"
    ],
)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Print startup message (basic logging without OTEL)
    print("Starting Weather Agent (NO OpenTelemetry)")
    print("This is a BLACK BOX - no observability!")
    print("UI: http://localhost:8000")
    
    # Launch Gradio server
    # server_port: Which port to listen on
    demo.launch(server_port=8000)
