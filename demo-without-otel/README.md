# Demo: Weather Agent WITHOUT OpenTelemetry

This demo shows a basic ADK agent with **no observability** - the "black box" approach.

## What This Demo Shows

- A working weather agent using Google ADK
- Interactive web interface via `adk web`
- **NO visibility** into internal operations

## The Problem

When you run this agent, you have NO way to know:
- ❌ How many tokens were used
- ❌ How long each LLM call took
- ❌ Which tools were called and why
- ❌ Where failures occurred
- ❌ The full request flow

This is the **"Black Box"** problem we'll solve in `demo-with-otel`.

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install google-adk google-generativeai litellm
```

### 3. Set Environment Variables

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Running the Demo

### Start the ADK Web Interface

```bash
cd demo-without-otel
adk web
```

### Open the Chat Interface

Open your browser to: **http://localhost:8000**

### Try These Queries

1. "What's the weather in Mumbai?"
2. "Give me a 5-day forecast for Bangalore"
3. "Compare weather in Delhi and Chennai"

## Project Structure

```
demo-without-otel/
├── README.md           # This file
└── weather_agent/
    ├── __init__.py     # Package marker
    └── agent.py        # Agent definition with tools
```

## Next Steps

After trying this demo, move to `demo-with-otel` to see how OpenTelemetry
transforms this "black box" into a "glass box" with full observability.
