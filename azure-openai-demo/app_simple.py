"""
Azure OpenAI Weather Agent with OpenTelemetry
Simplified version for Python 3.14 compatibility
"""

import os
import time
import re
import chainlit as cl
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -----------------------------------------------------------------------------
# Sensitive Data Redaction
# -----------------------------------------------------------------------------
SENSITIVE_PATTERNS = [
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***GOOGLE_API_KEY***'),
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***OPENAI_KEY***'),
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password|credential|auth)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), r'\1=***REDACTED***'),
]

def redact_sensitive(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "delhi": "28°C, Sunny, Air Quality: Moderate",
        "bangalore": "24°C, Pleasant, Light Rain Expected",
        "chennai": "34°C, Hot and Humid",
        "new york": "18°C, Clear, Breezy",
        "london": "12°C, Overcast, Light Drizzle",
        "tokyo": "22°C, Sunny, Humid",
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}. Available: {', '.join(weather_data.keys())}"

@tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a city."""
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
    return f"Forecast not available for {city}."

tools = [get_weather, get_forecast]

SYSTEM_PROMPT = """You are a helpful weather assistant powered by Azure OpenAI. You can:
1. Get current weather using the get_weather tool
2. Get forecasts using the get_forecast tool

Available cities: Mumbai, Delhi, Bangalore, Chennai, New York, London, Tokyo
Be concise and helpful."""

# -----------------------------------------------------------------------------
# Chainlit Handlers
# -----------------------------------------------------------------------------
@cl.on_chat_start
async def start():
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sre-resources.openai.azure.com")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    
    if not azure_api_key:
        await cl.Message(content="❌ AZURE_OPENAI_API_KEY not set").send()
        return
    
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        api_key=azure_api_key,
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    cl.user_session.set("llm", llm_with_tools)
    cl.user_session.set("messages", [SystemMessage(content=SYSTEM_PROMPT)])
    
    await cl.Message(content="👋 Hello! I'm your weather assistant powered by **Azure OpenAI**.\n\nAsk me about weather in Mumbai, Delhi, New York, London, or Tokyo!").send()

@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("llm")
    messages = cl.user_session.get("messages")
    
    if not llm:
        await cl.Message(content="❌ Agent not initialized").send()
        return
    
    # Add user message
    messages.append(HumanMessage(content=message.content))
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get LLM response
        response = await llm.ainvoke(messages)
        
        # Check if LLM wants to call tools
        if response.tool_calls:
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find and execute the tool
                if tool_name == "get_weather":
                    result = get_weather.invoke(tool_args)
                elif tool_name == "get_forecast":
                    result = get_forecast.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                tool_results.append(f"{tool_name}: {result}")
            
            # Add tool results to messages and get final response
            from langchain_core.messages import ToolMessage
            messages.append(response)
            for i, tool_call in enumerate(response.tool_calls):
                messages.append(ToolMessage(content=tool_results[i].split(": ", 1)[1], tool_call_id=tool_call["id"]))
            
            # Get final response after tool execution
            final_response = await llm.ainvoke(messages)
            final_content = final_response.content
            messages.append(final_response)
        else:
            final_content = response.content
            messages.append(response)
        
        cl.user_session.set("messages", messages)
        msg.content = final_content
        await msg.update()
        
    except Exception as e:
        safe_error = redact_sensitive(str(e))
        msg.content = f"❌ Error: {safe_error}"
        await msg.update()
