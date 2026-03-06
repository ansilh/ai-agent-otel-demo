# Instructor Notes: Observability of AI Agents - From Black Box to Glass Box

**Duration:** 60 minutes  
**Target Audience:** Engineering Students  
**Focus:** Observability & Hands-on Demo (Theory is foundational only)

---

## Session Overview & Time Allocation

| Section | Duration | Priority |
|---------|----------|----------|
| ML Basics (Quick Primer) | 8 min | Low - Foundation only |
| Applications & Inference | 7 min | Low - Foundation only |
| Data Access & Tool Calling | 5 min | Medium |
| Agentic AI Concepts | 5 min | Medium |
| **Demo: Building Agents with ADK** | 12 min | **High** |
| **Observability Concepts** | 8 min | **High** |
| **Demo: Integrating OTEL** | 12 min | **High** |
| Q&A / Buffer | 3 min | - |

---

## Section 1: ML Basics (8 minutes)

### 1.1 Data Structures (2 min)

**Talking Points:**
- **Array:** 1D collection of elements `[1, 2, 3, 4]`
- **Vector:** Array with direction/magnitude, used in embeddings
- **Tensor:** Multi-dimensional array (generalization)
  - Scalar (0D) → Vector (1D) → Matrix (2D) → Tensor (nD)

> 💡 **Key Message:** All ML operations are tensor math at scale

**[DIAGRAM PLACEHOLDER: tensor-dimensions.png]**
- Visual showing scalar → vector → matrix → 3D tensor progression

### 1.2 GPU Fundamentals (2 min)

**Talking Points:**
- CPU: Few powerful cores (sequential tasks)
- GPU: Thousands of smaller cores (parallel tasks)
- Matrix multiplication = highly parallelizable
- Why GPU for AI: Tensor operations run 10-100x faster

**[DIAGRAM PLACEHOLDER: cpu-vs-gpu-architecture.png]**
- Side-by-side comparison of CPU (4-8 cores) vs GPU (thousands of cores)

### 1.3 Weights & Biases (1 min)

**Talking Points:**
- **Weights:** Learnable parameters that determine connection strength
- **Biases:** Offset values added to outputs
- Together they form the "knowledge" of a model
- Training = adjusting weights & biases to minimize error

**Equation:**
```
y = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
```
Or: **y = W · X + b**

**Real-Life Example - House Price Prediction:**
```
Price = (w₁ × sqft) + (w₂ × bedrooms) + (w₃ × location_score) + bias
```

| Feature | Value | Weight | Contribution |
|---------|-------|--------|-------------|
| Square feet | 1500 | ₹5,000 | ₹75,00,000 |
| Bedrooms | 3 | ₹2,00,000 | ₹6,00,000 |
| Location score | 8 | ₹1,50,000 | ₹12,00,000 |
| Bias (base price) | - | - | ₹10,00,000 |
| **Predicted Price** | | | **₹1,03,00,000** |

**Key Teaching Points:**
- Weights = importance (higher weight for location means location matters most)
- Bias = starting point (base land value even with 0 features)
- Training = model learns optimal W & b from real data
- In LLMs: billions of weights capturing word/concept relationships

> 💡 **Interactive:** Ask "If location weight is highest, what does that tell us?"

### 1.4 AI vs ML (1 min)

**Talking Points:**
- **AI:** Broad field - machines mimicking human intelligence
- **ML:** Subset - learning from data without explicit programming
- **Deep Learning:** Subset of ML using neural networks
- **LLMs:** Deep learning models specialized for language

**[DIAGRAM PLACEHOLDER: ai-ml-dl-llm-hierarchy.png]**
- Nested circles: AI > ML > Deep Learning > LLMs

### 1.5 Model Types (1 min)

**Talking Points:**
- **LLM (Large Language Model):** Text generation (GPT, Gemini, Claude)
- **Vision Models:** Image understanding (CLIP, DALL-E)
- **Multimodal:** Text + Image + Audio (Gemini, GPT-4V)
- **Specialized:** Code (Codex), Speech (Whisper)

### 1.6 Parameters vs Hyperparameters (30 sec)

**Talking Points:**
- **Parameters:** Learned during training (weights, biases) - millions/billions
- **Hyperparameters:** Set before training (learning rate, batch size, temperature)

### 1.7 Model Building Basics (30 sec)

**Talking Points:**
- Data Collection → Preprocessing → Training → Evaluation → Deployment
- We focus on **inference** (using pre-trained models)

---

## Section 2: Applications & Inference (7 minutes)

### 2.1 Inferencing (1 min)

**Talking Points:**
- Using a trained model to make predictions
- Input → Model → Output
- No learning happens during inference
- This is what we do with APIs like Gemini

### 2.2 Tokens (2 min)

**Talking Points:**
- Text broken into chunks (not always words)
- "Hello world" → ["Hello", " world"] (2 tokens)
- "Tokenization" → ["Token", "ization"] (2 tokens)
- Pricing & limits based on tokens
- Typical: 1 token ≈ 4 characters in English

**[DIAGRAM PLACEHOLDER: tokenization-example.png]**
- Show a sentence being split into tokens with token IDs

### 2.3 Context & Context Window (1.5 min)

**Talking Points:**
- **Context:** All information the model sees in one request
- **Context Window:** Maximum tokens model can process
  - GPT-4: 128K tokens
  - Gemini 1.5: 1M+ tokens
- Larger context = more information but higher cost/latency

### 2.4 Memory & Hallucinations (1.5 min)

**Talking Points:**
- **Memory:** LLMs have no persistent memory between sessions
- Each request is independent (stateless)
- **Hallucinations:** Model generates plausible but false information
- Why? Model predicts likely tokens, not factual ones
- Solution: Grounding with real data (RAG, tools)

### 2.5 Prompting Techniques (1 min)

**Talking Points:**
- **Zero-shot:** Direct question, no examples
- **Few-shot:** Provide examples in prompt
- **Chain of Thought (CoT):** "Think step by step"
- **ReAct:** Reasoning + Acting (used in agents)

> 💡 **Key Message:** Prompting is how we "program" LLMs

---

## Section 3: Giving Data Access to LLMs (5 minutes)

### 3.1 Why Not Retraining? (2 min)

**Talking Points:**
- Retraining is expensive ($millions for large models)
- Time-consuming (weeks/months)
- Data becomes stale quickly
- Fine-tuning still limited for real-time data
- **Solution:** Give models access to external tools/data at runtime

### 3.2 Tool Calling / Function Calling (2 min)

**Talking Points:**
- Model decides when to call external functions
- Returns structured output (JSON) for function parameters
- Enables: Database queries, API calls, calculations, web search

**[DIAGRAM PLACEHOLDER: tool-calling-flow.png]**
- Flow: User Query → LLM → Tool Decision → Tool Execution → LLM → Response

### 3.3 MCP (Model Context Protocol) (1 min)

**Talking Points:**
- Anthropic's open standard for tool integration
- Standardized way for LLMs to interact with external systems
- Server-client architecture
- Growing ecosystem of MCP servers

---

## Section 4: Agentic AI (5 minutes)

### 4.1 What is an Agent? (2 min)

**Talking Points:**
- LLM + Tools + Autonomy + Goal-directed behavior
- Can plan, execute, and iterate
- Makes decisions about which tools to use
- Can handle multi-step tasks

> 💡 **Key Message:** Agent = LLM that can take actions, not just generate text

**[DIAGRAM PLACEHOLDER: agent-architecture.png]**
- Components: LLM Brain, Tools, Memory, Planning Module

### 4.2 Agent Patterns (2 min)

**Talking Points:**
- **ReAct:** Reason → Act → Observe → Repeat
- **Plan-and-Execute:** Create plan first, then execute steps
- **Multi-Agent:** Multiple specialized agents collaborating
- **Hierarchical:** Manager agent delegates to worker agents

**[DIAGRAM PLACEHOLDER: agent-patterns.png]**
- Visual comparison of ReAct loop vs Multi-agent collaboration

### 4.3 Frameworks for Building Agents (1 min)

**Talking Points:**
- **Google ADK (Agent Development Kit):** What we'll use today
- **LangChain/LangGraph:** Popular Python framework
- **CrewAI:** Multi-agent orchestration
- **AutoGen:** Microsoft's multi-agent framework

---

## Section 5: DEMO - Building Agents with Google ADK (12 minutes)

### Setup Checklist (Before Session)
- [ ] Python 3.10+ installed
- [ ] Google Cloud account with Gemini API access
- [ ] ADK installed: `pip install google-adk`
- [ ] API key configured
- [ ] SigNoz running locally (for later demo)

### Demo 5.1: Single Agent - Single Action (5 min)

**What to Show:**
1. Create a simple agent with one tool (e.g., weather lookup)
2. Show the agent receiving a query
3. Show the agent deciding to use the tool
4. Show the tool execution and response

**Code Location:** `demos/single-agent/`

**Talking Points During Demo:**
- Point out the tool definition
- Show how agent decides to call the tool
- Highlight the structured output

### Demo 5.2: Multi-Agent Example (7 min)

**What to Show:**
1. Create two agents with different capabilities
2. Show agent orchestration/delegation
3. Demonstrate agents collaborating on a task

**Code Location:** `demos/multi-agent/`

**Talking Points During Demo:**
- Each agent has specialized tools
- Orchestrator decides which agent to invoke
- Show the conversation flow between agents

> ⚠️ **Transition:** "Now we have working agents, but how do we know what's happening inside? This is where observability comes in."

---

## Section 6: Observability Concepts (8 minutes)

### 6.1 Why Observability for Agents? (2 min)

**Talking Points:**
- Agents are non-deterministic (same input ≠ same output)
- Multiple LLM calls, tool invocations, decision points
- Need to understand: What happened? Why? How long?
- Debugging without observability = guessing

> 💡 **Key Message:** Observability turns the "black box" into a "glass box"

### 6.2 OpenTelemetry (OTEL) (2 min)

**Talking Points:**
- Open standard for observability
- Vendor-neutral instrumentation
- Three pillars: Traces, Metrics, Logs
- Wide ecosystem support

**[DIAGRAM PLACEHOLDER: otel-architecture.png]**
- Components: SDK → Collector → Backend (SigNoz)

### 6.3 The Three Signals (3 min)

**Traces:**
- End-to-end request journey
- Spans represent operations (LLM call, tool execution)
- Parent-child relationships show flow
- **Critical for agents:** See every step the agent took

**Metrics:**
- Numerical measurements over time
- Token usage, latency, error rates
- Aggregated data for dashboards

**Logs:**
- Discrete events with context
- Prompts, responses, errors
- Detailed debugging information

**[DIAGRAM PLACEHOLDER: three-signals.png]**
- Visual showing Traces (timeline), Metrics (graphs), Logs (events)

### 6.4 OTEL-Compatible Backends (1 min)

**Talking Points:**
- **SigNoz:** Open-source, what we'll use
- **Jaeger:** Tracing-focused
- **Grafana/Tempo:** Popular stack
- **Commercial:** Datadog, New Relic, Honeycomb

---

## Section 7: DEMO - Integrating OTEL in Agents (12 minutes)

### Pre-Demo: SigNoz Setup (Show briefly, should be pre-installed)

```bash
# Docker Compose installation (done before session)
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker-compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

**SigNoz UI:** http://localhost:3301

### Demo 7.1: Adding OTEL to Single Agent (5 min)

**What to Show:**
1. Add OTEL dependencies
2. Configure tracer provider
3. Instrument LLM calls
4. Instrument tool calls
5. Run agent and show traces in SigNoz

**Key Code Changes to Highlight:**
```python
# Tracer setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create spans for each operation
with tracer.start_as_current_span("llm_call") as span:
    span.set_attribute("model", "gemini-pro")
    span.set_attribute("prompt_tokens", token_count)
    # ... LLM call
```

**Talking Points During Demo:**
- Show span hierarchy in SigNoz
- Point out timing information
- Show custom attributes (tokens, model, etc.)

### Demo 7.2: Tracing Multi-Agent Interactions (5 min)

**What to Show:**
1. Trace propagation between agents
2. Visualize agent-to-agent communication
3. Show complete request flow across agents

**Talking Points During Demo:**
- Parent trace spans entire request
- Child spans for each agent's work
- Can see exactly where time is spent
- Identify bottlenecks and failures

### Demo 7.3: Analyzing Agent Behavior (2 min)

**What to Show in SigNoz:**
1. Filter traces by agent name
2. Show latency distribution
3. Identify slow operations
4. Show error traces

> 💡 **Key Message:** "Now you can see exactly what your agent is thinking and doing"

---

## Section 8: Q&A / Wrap-up (3 minutes)

### Key Takeaways

1. **Agents = LLM + Tools + Autonomy**
2. **Observability is essential** for understanding agent behavior
3. **OpenTelemetry** provides standardized instrumentation
4. **Three signals:** Traces (flow), Metrics (numbers), Logs (events)
5. **SigNoz** is a great open-source option for visualization

### Resources to Share

- Google ADK Documentation: https://google.github.io/adk-docs/
- OpenTelemetry: https://opentelemetry.io/
- SigNoz: https://signoz.io/
- Session Code: [Your GitHub repo]

### Questions to Anticipate

1. "How does this compare to LangSmith?" → LangSmith is LangChain-specific, OTEL is universal
2. "What about cost tracking?" → Can add token counts as span attributes
3. "Does this add latency?" → Minimal (<1ms per span), async export
4. "Can I use this in production?" → Yes, OTEL is production-ready

---

## Backup Content (If Time Permits)

### Advanced Tracing Patterns
- Semantic conventions for GenAI (emerging standard)
- Custom span processors for sensitive data redaction
- Sampling strategies for high-volume agents

### Metrics Deep Dive
- Token usage dashboards
- Cost tracking per agent
- Error rate monitoring

---

## Technical Setup Checklist

### Before the Session
- [ ] Test all demos end-to-end
- [ ] Verify SigNoz is running and accessible
- [ ] Pre-load API keys in environment
- [ ] Have backup recordings of demos
- [ ] Test screen sharing and resolution

### Required Dependencies
```bash
pip install google-adk
pip install opentelemetry-api
pip install opentelemetry-sdk
pip install opentelemetry-exporter-otlp
pip install opentelemetry-instrumentation
```

### Environment Variables
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="agent-demo"
```

---

## Diagram Placeholders Summary

| Diagram Name | Description | Section |
|--------------|-------------|---------|
| tensor-dimensions.png | Scalar → Vector → Matrix → Tensor | 1.1 |
| cpu-vs-gpu-architecture.png | CPU vs GPU core comparison | 1.2 |
| ai-ml-dl-llm-hierarchy.png | Nested circles showing hierarchy | 1.4 |
| tokenization-example.png | Sentence split into tokens | 2.2 |
| tool-calling-flow.png | LLM tool calling sequence | 3.2 |
| agent-architecture.png | Agent components diagram | 4.1 |
| agent-patterns.png | ReAct vs Multi-agent patterns | 4.2 |
| otel-architecture.png | OTEL SDK → Collector → Backend | 6.2 |
| three-signals.png | Traces, Metrics, Logs visual | 6.3 |
