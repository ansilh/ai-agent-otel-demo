# Demo: Weather Agent using Chainlit WITHOUT OpenTelemetry

This demo shows a beginner-friendly agent using **Chainlit** for a beautiful web UI.
It demonstrates the "black box" problem - no observability.

## Why Chainlit?

| Feature | Benefit |
|---------|---------|
| **Beautiful UI** | Modern chat interface out of the box |
| **Beginner-friendly** | Simple decorator-based API |
| **Local LLM support** | Native Ollama integration |
| **Minimal code** | ~100 lines for a working agent |

## The Problem

When you run this agent, you have NO way to know:
- ❌ How many tokens were used
- ❌ How long each LLM call took
- ❌ Which tools were called and why
- ❌ Where failures occurred
- ❌ The full request flow

This is the **"Black Box"** problem we'll solve in `chainlit-with-otel`.

## Setup

### 1. Install and Start Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen model (choose based on your RAM)
ollama pull qwen2.5:7b    # ~8GB RAM needed
# ollama pull qwen2.5:14b  # ~16GB RAM needed
# ollama pull qwen2.5:32b  # ~40GB RAM needed

# Start Ollama server
ollama serve
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd chainlit-without-otel
pip install -r requirements.txt
```

## Running the Demo

```bash
chainlit run app.py
```

This will:
1. Start the Chainlit server
2. Open your browser to **http://localhost:8000**
3. Show a beautiful chat interface

## Try These Queries

1. "What's the weather in Mumbai?"
2. "Give me a 5-day forecast for Bangalore"
3. "Compare weather in Delhi and Chennai"

## Project Structure

```
chainlit-without-otel/
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── app.py              # Main application (single file!)
```

## Code Highlights

### Simple Tool Definition
```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # ... implementation
```

### Easy Chat Handler
```python
@cl.on_message
async def main(message: cl.Message):
    response = await agent_executor.ainvoke({"input": message.content})
    await cl.Message(content=response["output"]).send()
```

## Next Steps

After trying this demo, move to `chainlit-with-otel` to see how OpenTelemetry
transforms this "black box" into a "glass box" with full observability.
