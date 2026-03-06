# Slide Deck: Observability of AI Agents - From Black Box to Glass Box

**Total Slides:** ~35-40  
**Duration:** 60 minutes  
**Format:** Each slide entry includes content and speaker notes

---

## Title Slide

### Slide 1: Title
**Content:**
- **Title:** Observability of AI Agents
- **Subtitle:** From Black Box to Glass Box
- **Presenter:** [Your Name]
- **Date:** [Session Date]
- **Logo/Affiliation:** [Your Organization]

**Speaker Notes:** Welcome, introduce yourself, set expectations for the session.

---

## Section 1: ML Basics (8 min | ~6 slides)

### Slide 2: Section Header - ML Basics
**Content:**
- **Title:** ML Basics
- **Subtitle:** Quick Foundation for Understanding Agents

---

### Slide 3: Data Structures in ML
**Content:**
- **Array:** `[1, 2, 3, 4]` - 1D collection
- **Vector:** Array with meaning (embeddings)
- **Matrix:** 2D array
- **Tensor:** n-dimensional array

**[DIAGRAM PLACEHOLDER: tensor-dimensions.png]**
*Visual: Scalar (0D) → Vector (1D) → Matrix (2D) → 3D Tensor*

**Speaker Notes:** All ML is tensor math. Models process tensors, output tensors.

---

### Slide 4: Why GPUs?
**Content:**
| CPU | GPU |
|-----|-----|
| 4-16 powerful cores | 1000s of smaller cores |
| Sequential processing | Parallel processing |
| General purpose | Matrix math optimized |

**[DIAGRAM PLACEHOLDER: cpu-vs-gpu-architecture.png]**
*Visual: Side-by-side core comparison*

**Speaker Notes:** Matrix multiplication is embarrassingly parallel - perfect for GPUs.

---

### Slide 5: Weights, Biases & Learning
**Content:**
- **Weights:** Connection strengths (learned)
- **Biases:** Offset values (learned)
- **Training:** Adjusting W & B to minimize error
- **Result:** The "knowledge" of the model

**Equation:**
```
y = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
```
Or in vector form: **y = W · X + b**

Where:
- `y` = output (prediction)
- `W` = weights (how important each input is)
- `X` = inputs (features)
- `b` = bias (baseline offset)

**Speaker Notes:** Billions of parameters = billions of weights & biases.

---

### Slide 5b: Real-Life Example - House Price Prediction
**Content:**
**Predicting House Price:**
```
Price = (w₁ × sqft) + (w₂ × bedrooms) + (w₃ × location_score) + b
```

| Feature | Value | Weight | Contribution |
|---------|-------|--------|-------------|
| Square feet | 1500 | ₹5,000 | ₹75,00,000 |
| Bedrooms | 3 | ₹2,00,000 | ₹6,00,000 |
| Location score | 8 | ₹1,50,000 | ₹12,00,000 |
| **Bias** (base price) | - | - | ₹10,00,000 |
| **Predicted Price** | | | **₹1,03,00,000** |

**Key Insight:**
- **Weights tell us importance:** sqft weight (₹5,000) > bedroom weight shows sqft matters more
- **Bias is the starting point:** Even a 0 sqft, 0 bedroom house has a base land value
- **Training adjusts these:** Model learns optimal W & b from thousands of real sales

**Speaker Notes:** 
- Ask audience: "If location_score weight is highest, what does that tell us?"
- Answer: Location is the most important factor in this model
- In LLMs: Instead of 3 weights, we have **billions** - each capturing relationships between words, concepts, patterns

---

### Slide 5c: Word Embeddings - Distance Between Words
**Content:**
**How LLMs "Understand" Words:**
- Words converted to **vectors** (list of numbers)
- Similar words → vectors close together
- Different words → vectors far apart

**Example Embeddings (simplified 3D):**
```
"king"   → [0.8, 0.2, 0.9]
"queen"  → [0.7, 0.3, 0.9]
"apple"  → [0.1, 0.9, 0.2]
```

**Distance = Similarity:**
- king ↔ queen: **Close** (both royalty)
- king ↔ apple: **Far** (unrelated)

**[DIAGRAM PLACEHOLDER: word-embeddings-3d.png]**
*Visual: 3D space with king/queen clustered, apple far away, man/woman clustered*

**Famous Example:**
```
king - man + woman ≈ queen
```
Vector math captures semantic relationships!

**Speaker Notes:**
- Embeddings are learned during training
- Real embeddings have 768-4096 dimensions, not 3
- This is how models "know" that cat and dog are related

---

### Slide 5d: How LLMs Predict the Next Word
**Content:**
**The Core Task:** Given previous words, predict the next word

**Input:** "The cat sat on the ___"

**Process:**
1. Convert words to embeddings (vectors)
2. Pass through layers of **weights & biases**
3. Each layer transforms and combines information
4. Output: probability for every possible next word

**[DIAGRAM PLACEHOLDER: next-word-prediction.png]**
*Visual: Input tokens → Transformer layers → Probability distribution*

**Output Probabilities:**
| Word | Probability |
|------|-------------|
| mat | 35% |
| floor | 25% |
| couch | 15% |
| roof | 10% |
| ... | ... |

**Role of Weights & Biases:**
- **Weights:** Determine which word relationships matter
  - High weight between "sat" and "mat" (common phrase)
- **Biases:** Shift probabilities based on learned patterns
  - Bias toward common words like "the", "a"

**Speaker Notes:**
- Billions of weights = billions of learned relationships
- Training: Adjust W & B so "mat" gets high probability after "cat sat on the"
- The model learned this from seeing millions of sentences

---

### Slide 6: AI Landscape
**Content:**

**[DIAGRAM PLACEHOLDER: ai-ml-dl-llm-hierarchy.png]**
*Visual: Nested circles - AI > ML > Deep Learning > LLMs*

- **AI:** Machines mimicking intelligence
- **ML:** Learning from data
- **Deep Learning:** Neural networks
- **LLMs:** Language-specialized DL

**Speaker Notes:** We're working with LLMs today - the innermost circle.

---

### Slide 7: Parameters vs Hyperparameters
**Content:**
| Parameters | Hyperparameters |
|------------|-----------------|
| Learned during training | Set before training |
| Weights & Biases | Learning rate, temperature |
| Millions/Billions | Dozens |
| Model's knowledge | Training configuration |

**Speaker Notes:** We don't train models - we use them. We adjust hyperparameters like temperature.

---

## Section 2: Applications & Inference (7 min | ~5 slides)

### Slide 8: Section Header - Applications
**Content:**
- **Title:** LLM Applications
- **Subtitle:** How We Use Pre-trained Models

---

### Slide 9: Inference
**Content:**
```
Input → [Trained Model] → Output
```

- No learning during inference
- Using the model's existing knowledge
- What happens when you call an API

**Speaker Notes:** Training costs millions. Inference costs cents. We do inference.

---

### Slide 10: Tokens
**Content:**
```
"Hello world" → ["Hello", " world"] → [15496, 995]
```

- Text split into chunks (not always words)
- ~1 token ≈ 4 characters (English)
- Pricing & limits based on tokens

**[DIAGRAM PLACEHOLDER: tokenization-example.png]**
*Visual: Sentence → Token splits → Token IDs*

**Speaker Notes:** "Tokenization" becomes 2 tokens. Important for cost and context limits.

---

### Slide 11: Context & Context Window
**Content:**
- **Context:** Everything the model sees in one request
- **Context Window:** Maximum capacity

| Model | Context Window |
|-------|----------------|
| GPT-4 | 128K tokens |
| Gemini 1.5 | 1M+ tokens |
| Claude 3 | 200K tokens |

**Speaker Notes:** Larger context = more info but higher cost and latency.

---

### Slide 12: Memory & Hallucinations
**Content:**
- **No Persistent Memory:** Each request is independent
- **Hallucinations:** Plausible but false outputs

**Why hallucinations?**
- Model predicts *likely* tokens
- Not *factual* tokens

**Solution:** Ground with real data → **Tools & RAG**

**Speaker Notes:** This is why we need tool calling - to ground responses in reality.

---

### Slide 13: Prompting Techniques
**Content:**
| Technique | Description |
|-----------|-------------|
| **Zero-shot** | Direct question, no examples |
| **Few-shot** | Include examples in prompt |
| **Chain of Thought** | "Think step by step" |
| **ReAct** | Reasoning + Acting (agents!) |

**Speaker Notes:** ReAct is the foundation of agentic behavior.

---

## Section 3: Data Access & Tool Calling (5 min | ~4 slides)

### Slide 14: Section Header - Data Access
**Content:**
- **Title:** Giving LLMs Data Access
- **Subtitle:** Beyond Static Training Data

---

### Slide 15: Why Not Retrain?
**Content:**
| Challenge | Reality |
|-----------|---------|
| Cost | $Millions for large models |
| Time | Weeks to months |
| Freshness | Data stale immediately |
| Expertise | Requires ML teams |

**Better Approach:** Runtime data access via **Tools**

**Speaker Notes:** We give models tools instead of retraining them.

---

### Slide 16: Tool Calling / Function Calling
**Content:**

**[DIAGRAM PLACEHOLDER: tool-calling-flow.png]**
*Visual: User → LLM → Tool Decision → Tool Execution → LLM → Response*

- Model decides **when** to call tools
- Returns structured JSON for parameters
- Enables: APIs, databases, calculations, search

**Speaker Notes:** The model becomes a reasoning engine that orchestrates tools.

---

### Slide 17: MCP (Model Context Protocol)
**Content:**
- **What:** Open standard by Anthropic
- **Why:** Standardized tool integration
- **How:** Server-client architecture

```
LLM ←→ MCP Client ←→ MCP Server ←→ External Systems
```

**Speaker Notes:** Growing ecosystem - many pre-built MCP servers available.

---

## Section 4: Agentic AI (5 min | ~4 slides)

### Slide 18: Section Header - Agentic AI
**Content:**
- **Title:** Agentic AI
- **Subtitle:** LLMs That Take Action

---

### Slide 19: What is an Agent?
**Content:**

**[DIAGRAM PLACEHOLDER: agent-architecture.png]**
*Visual: LLM Brain + Tools + Memory + Planning*

**Agent = LLM + Tools + Autonomy + Goals**

- Plans and executes multi-step tasks
- Decides which tools to use
- Iterates based on results

**Speaker Notes:** Key difference from chatbots - agents ACT, not just respond.

---

### Slide 20: Agent Patterns
**Content:**

**[DIAGRAM PLACEHOLDER: agent-patterns.png]**
*Visual: ReAct loop + Multi-agent collaboration*

| Pattern | Description |
|---------|-------------|
| **ReAct** | Reason → Act → Observe → Repeat |
| **Plan-Execute** | Plan first, then execute |
| **Multi-Agent** | Specialized agents collaborate |
| **Hierarchical** | Manager delegates to workers |

**Speaker Notes:** We'll demo ReAct and Multi-Agent patterns.

---

### Slide 21: Agent Frameworks
**Content:**
| Framework | Highlights |
|-----------|------------|
| **Google ADK** | What we'll use today |
| LangChain/LangGraph | Popular, flexible |
| CrewAI | Multi-agent focused |
| AutoGen | Microsoft, multi-agent |

**Speaker Notes:** ADK is new, clean API, great Gemini integration.

---

## Section 5: DEMO - Building Agents (12 min | ~3 slides)

### Slide 22: Section Header - Demo Time
**Content:**
- **Title:** Demo: Building Agents
- **Subtitle:** Google ADK + Gemini

---

### Slide 23: Demo 1 - Single Agent
**Content:**
**What We'll Build:**
- One agent with one tool (weather lookup)
- Agent receives query
- Agent decides to use tool
- Tool executes, agent responds

```python
# Live coding / demo
```

**Speaker Notes:** Show tool definition, agent decision, structured output.

---

### Slide 24: Demo 2 - Multi-Agent
**Content:**
**What We'll Build:**
- Two specialized agents
- Orchestrator for delegation
- Agents collaborate on task

```python
# Live coding / demo
```

**Speaker Notes:** Show agent-to-agent communication, task delegation.

---

## Section 6: Observability Concepts (8 min | ~6 slides)

### Slide 25: Section Header - Observability
**Content:**
- **Title:** Observability in Agents
- **Subtitle:** From Black Box to Glass Box 🔍

---

### Slide 26: Why Observability for Agents?
**Content:**
**The Problem:**
- Agents are **non-deterministic**
- Multiple LLM calls, tool invocations
- Complex decision trees
- Same input ≠ same output

**Without observability = Debugging by guessing**

**Speaker Notes:** This is the core message - you NEED visibility into agent behavior.

---

### Slide 27: OpenTelemetry (OTEL)
**Content:**

**[DIAGRAM PLACEHOLDER: otel-architecture.png]**
*Visual: App + SDK → Collector → Backend (SigNoz)*

- **Open standard** for observability
- **Vendor-neutral** instrumentation
- **Three pillars:** Traces, Metrics, Logs
- **Wide ecosystem** support

**Speaker Notes:** OTEL is the industry standard - learn once, use everywhere.

---

### Slide 28: Signal 1 - Traces (Deep Dive)
**Content:**

**What is a Trace?**
- Complete journey of a request through your system
- Made up of **Spans** (individual operations)

**Trace Structure:**
```
Trace ID: abc123
│
├── Span: "agent_request" (parent) ─────────── 1200ms
│   ├── Span: "llm_call_1" (child) ─────────── 450ms
│   │   └── Attributes: model=gemini, tokens=150
│   ├── Span: "tool_weather_api" (child) ───── 120ms
│   │   └── Attributes: city=Mumbai, status=200
│   └── Span: "llm_call_2" (child) ─────────── 380ms
│       └── Attributes: model=gemini, tokens=85
```

**Span Anatomy:**
| Component | Description | Example |
|-----------|-------------|---------|
| **Trace ID** | Unique ID for entire request | `abc123` |
| **Span ID** | Unique ID for this operation | `span_456` |
| **Parent Span ID** | Links to parent span | `span_123` |
| **Name** | Operation name | `llm_call` |
| **Start/End Time** | Duration tracking | `450ms` |
| **Attributes** | Key-value metadata | `model=gemini` |
| **Status** | OK, ERROR | `OK` |
| **Events** | Timestamped logs within span | `token_count: 150` |

**[DIAGRAM PLACEHOLDER: trace-span-anatomy.png]**
*Visual: Waterfall view showing parent span with nested child spans, timing bars*

**Speaker Notes:** 
- Trace = the whole story, Span = one chapter
- Parent-child shows causality (which operation triggered which)
- Attributes are crucial - add model name, token counts, tool names
- This is the MOST important signal for debugging agents

---

### Slide 29: Signal 2 - Metrics (Deep Dive)
**Content:**

**What are Metrics?**
- Numerical measurements aggregated over time
- Statistical data for dashboards & alerts

**Metric Types:**
| Type | Description | Example |
|------|-------------|---------|
| **Counter** | Only increases | Total requests: 1000 |
| **Gauge** | Can go up/down | Active connections: 42 |
| **Histogram** | Distribution of values | Latency p50, p95, p99 |

**Agent-Specific Metrics:**
```
# Token Usage
agent_tokens_total{type="input"} 15000
agent_tokens_total{type="output"} 8500

# Latency
agent_request_duration_seconds{quantile="0.5"} 1.2
agent_request_duration_seconds{quantile="0.95"} 3.5

# Tool Calls
agent_tool_calls_total{tool="weather_api"} 250
agent_tool_calls_total{tool="search"} 180

# Errors
agent_errors_total{type="llm_timeout"} 12
```

**[DIAGRAM PLACEHOLDER: metrics-dashboard.png]**
*Visual: Dashboard with token usage graph, latency percentiles, error rate*

**Speaker Notes:**
- Metrics answer: How much? How fast? How often?
- Use for alerting (e.g., error rate > 5%)
- Aggregated = good for trends, bad for debugging single requests

---

### Slide 30: Signal 3 - Logs (Deep Dive)
**Content:**

**What are Logs?**
- Discrete, timestamped events
- Detailed context for debugging

**Structured Log Example:**
```json
{
  "timestamp": "2024-03-06T10:30:45Z",
  "level": "INFO",
  "trace_id": "abc123",
  "span_id": "span_456",
  "message": "LLM call completed",
  "attributes": {
    "model": "gemini-pro",
    "prompt_tokens": 150,
    "completion_tokens": 85,
    "latency_ms": 450,
    "prompt": "What is the weather in Mumbai?",
    "response": "The weather in Mumbai is..."
  }
}
```

**Agent Log Levels:**
| Level | When to Use | Example |
|-------|-------------|---------|
| **DEBUG** | Detailed tracing | Full prompt/response |
| **INFO** | Normal operations | "Tool called: weather_api" |
| **WARN** | Potential issues | "Retry attempt 2 of 3" |
| **ERROR** | Failures | "LLM timeout after 30s" |

**Key Fields for Agent Logs:**
- `trace_id` - Link to trace
- `prompt` / `response` - What was said
- `tool_name` / `tool_input` / `tool_output`
- `error_message` / `stack_trace`

**Speaker Notes:**
- Always include trace_id to correlate with traces
- Be careful with sensitive data in prompts/responses
- Structured logs (JSON) > plain text for querying

---

### Slide 31: Three Signals Together
**Content:**

**[DIAGRAM PLACEHOLDER: three-signals.png]**
*Visual: Traces (timeline) + Metrics (graphs) + Logs (events)*

| Signal | Question Answered |
|--------|-------------------|
| Traces | What happened? In what order? |
| Metrics | How much? How fast? How often? |
| Logs | What exactly was said/done? |

**Speaker Notes:** Use all three together for complete observability.

---

### Slide 32: OTEL Backends
**Content:**
| Backend | Type |
|---------|------|
| **SigNoz** | Open-source (our choice) |
| Jaeger | Open-source, tracing |
| Grafana + Tempo | Open-source stack |
| Datadog | Commercial |
| Honeycomb | Commercial |

**Speaker Notes:** SigNoz is free, full-featured, easy to run locally.

---

## Section 7: DEMO - OTEL Integration (12 min | ~4 slides)

### Slide 33: Section Header - OTEL Demo
**Content:**
- **Title:** Demo: Adding Observability
- **Subtitle:** Instrumenting Our Agents

---

### Slide 34: SigNoz Setup
**Content:**
```bash
# Pre-installed for demo
docker-compose up -d
```

**UI:** http://localhost:3301

**Speaker Notes:** Show SigNoz UI briefly - traces, metrics, logs tabs.

---

### Slide 35: Demo 3 - Instrumenting Single Agent
**Content:**
**What We'll Add:**
```python
from opentelemetry import trace
tracer = trace.get_tracer("agent-demo")

with tracer.start_as_current_span("llm_call") as span:
    span.set_attribute("model", "gemini-pro")
    span.set_attribute("tokens", count)
    # ... LLM call
```

**Speaker Notes:** Show span hierarchy in SigNoz, timing, attributes.

---

### Slide 36: Demo 4 - Multi-Agent Tracing
**Content:**
**What We'll See:**
- Trace spans entire request
- Child spans per agent
- Tool execution timing
- Complete request flow

**Speaker Notes:** Show trace propagation, identify bottlenecks.

---

### Slide 37: Analyzing in SigNoz
**Content:**
**Live Demo:**
- Filter traces by agent
- Latency distribution
- Error identification
- Span details

**Speaker Notes:** This is the "glass box" - full visibility into agent behavior.

---

## Closing (3 min | ~3 slides)

### Slide 38: Key Takeaways
**Content:**
1. **Agents = LLM + Tools + Autonomy**
2. **Observability is essential** for agent development
3. **OpenTelemetry** = industry standard
4. **Three signals:** Traces, Metrics, Logs
5. **SigNoz** = great open-source option

**Speaker Notes:** Reinforce main points.

---

### Slide 39: Resources
**Content:**
- **Google ADK:** https://google.github.io/adk-docs/
- **OpenTelemetry:** https://opentelemetry.io/
- **SigNoz:** https://signoz.io/
- **Session Code:** [GitHub Repo URL]

**Speaker Notes:** Share links, mention code is available.

---

### Slide 40: Q&A
**Content:**
- **Title:** Questions?
- **Contact:** [Your Email/Social]

**Speaker Notes:** Open floor for questions.

---

## Diagram Placeholders Summary

| Slide | Diagram File | Description |
|-------|--------------|-------------|
| 3 | tensor-dimensions.png | Scalar → Vector → Matrix → Tensor progression |
| 4 | cpu-vs-gpu-architecture.png | CPU vs GPU core comparison |
| 5c | word-embeddings-3d.png | 3D space with king/queen clustered, apple far away |
| 5d | next-word-prediction.png | Input tokens → Transformer layers → Probability distribution |
| 6 | ai-ml-dl-llm-hierarchy.png | Nested circles: AI > ML > DL > LLM |
| 10 | tokenization-example.png | Sentence → Tokens → Token IDs |
| 16 | tool-calling-flow.png | User → LLM → Tool → LLM → Response |
| 19 | agent-architecture.png | Agent components (LLM, Tools, Memory, Planning) |
| 20 | agent-patterns.png | ReAct loop + Multi-agent visual |
| 27 | otel-architecture.png | App → SDK → Collector → Backend |
| 28 | trace-span-anatomy.png | Waterfall view with parent/child spans, timing bars |
| 29 | metrics-dashboard.png | Dashboard with token usage, latency percentiles, error rate |
| 31 | three-signals.png | Traces + Metrics + Logs combined view |

---

## Slide Design Recommendations

**Color Scheme:**
- Primary: Deep blue (#1a365d) - trust, technology
- Accent: Bright green (#38a169) - observability, visibility
- Background: Light gray (#f7fafc) or dark mode

**Typography:**
- Headings: Bold sans-serif (Inter, Roboto)
- Body: Clean sans-serif
- Code: Monospace (Fira Code, JetBrains Mono)

**Visual Style:**
- Minimal text per slide
- Large diagrams
- Code snippets with syntax highlighting
- Consistent iconography

**Tools for Creation:**
- Google Slides / PowerPoint
- Figma (for diagrams)
- Excalidraw (for hand-drawn style diagrams)
- Carbon (for code screenshots)
