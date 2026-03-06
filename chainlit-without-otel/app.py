"""
=============================================================================
DEMO: Weather Agent using Chainlit WITHOUT OpenTelemetry
=============================================================================

This is a beginner-friendly agent using Chainlit for a beautiful web UI.
It demonstrates the "black box" problem - the agent works, but we have
no visibility into what's happening inside.

Run with: chainlit run app.py
Then open http://localhost:8000 to interact with the agent.
=============================================================================
"""

# Chainlit provides the web chat interface
import chainlit as cl

# LangChain components for building the agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor


# -----------------------------------------------------------------------------
# Define Tools
# -----------------------------------------------------------------------------
# Tools are functions that the agent can decide to call based on user queries.
# The @tool decorator makes them available to the LangChain agent.

@tool
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
# Create the Agent
# -----------------------------------------------------------------------------

# List of tools the agent can use
tools = [get_weather, get_forecast]

# System prompt that defines the agent's behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful weather assistant. You can:

1. Get current weather for cities in India using the get_weather tool
2. Get weather forecasts using the get_forecast tool

When users ask about weather:
- Use get_weather for current conditions
- Use get_forecast for future predictions
- Be friendly and provide helpful context about the weather

If asked about cities not in your database, let the user know which cities are available.
Always be concise but informative in your responses."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# -----------------------------------------------------------------------------
# Chainlit Event Handlers
# -----------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    """
    Called when a new chat session starts.
    Initialize the LLM and agent here.
    """
    # Create the LLM - using Qwen via Ollama (local, no rate limits!)
    # Change model to "qwen2.5:14b" or "qwen2.5:32b" for better quality
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",  # Ollama server
    )
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the executor that runs the agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Print agent's thinking to console
    )
    
    # Store in session for use in message handler
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_history", [])
    
    # Welcome message
    await cl.Message(
        content="👋 Hello! I'm your weather assistant. Ask me about weather in Indian cities like Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, or Pune!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Called when the user sends a message.
    Run the agent and return the response.
    """
    # Get agent and history from session
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history")
    
    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Run the agent
    response = await agent_executor.ainvoke({
        "input": message.content,
        "chat_history": chat_history,
    })
    
    # Update chat history
    chat_history.append(("human", message.content))
    chat_history.append(("assistant", response["output"]))
    cl.user_session.set("chat_history", chat_history)
    
    # Send the response
    msg.content = response["output"]
    await msg.update()


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
# In the next demo (chainlit-with-otel), we'll add OpenTelemetry
# to turn this into a "GLASS BOX" where we can see everything.
# =============================================================================
