# Security Best Practices for AI Agent Observability

## ⚠️ CRITICAL: Never Log Sensitive Data

When implementing observability for AI agents, **NEVER** log the following in traces, metrics, or logs:

- API keys (Google, OpenAI, Azure, etc.)
- Bearer tokens
- Passwords or credentials
- Authorization headers
- URLs with embedded credentials or API keys

## The Problem

Error messages from LLM providers often contain the full URL including API keys:

```
# BAD - This exposes your API key!
Error: 404 Not Found for url 'https://api.example.com/v1/chat?key=AIzaSyABC123...'
```

## The Solution

All demos in this repository use a `redact_sensitive()` function that automatically removes sensitive data before logging:

```python
# Patterns we detect and redact:
- Google API keys (AIza...)
- OpenAI keys (sk-...)
- Bearer tokens
- Authorization headers
- URL query parameters with keys/tokens
- Generic secrets and passwords
```

## How to Use

### 1. Import the redaction functions

```python
from redaction import redact_sensitive, safe_str
```

### 2. Always redact before logging

```python
# BAD - May expose secrets
logger.error(f"Error: {error}")
span.set_attribute("error.message", str(error))

# GOOD - Secrets are redacted
logger.error(f"Error: {redact_sensitive(str(error))}")
span.set_attribute("error.message", redact_sensitive(str(error)))
```

### 3. Use safe_str() for unknown objects

```python
# Converts to string, redacts, and truncates
span.set_attribute("tool.output", safe_str(result))
```

### 4. Never use record_exception() without review

```python
# BAD - Full stack trace may contain secrets
span.record_exception(e)

# GOOD - Only record safe attributes
span.set_attribute("error.type", type(e).__name__)
span.set_attribute("error.message", redact_sensitive(str(e)))
```

## Checklist for Secure Observability

- [ ] All error messages are redacted before logging
- [ ] All span attributes use `safe_str()` or `redact_sensitive()`
- [ ] No `record_exception()` calls without review
- [ ] Environment variables with keys are never logged
- [ ] URLs in logs have query parameters redacted
- [ ] User-facing error messages are also redacted

## Testing Your Redaction

```python
# Test that your patterns work
test_cases = [
    "API key is AIzaSyABC123456789012345678901234567890",
    "Bearer sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890",
    "https://api.example.com?api_key=secret123&other=value",
]

for test in test_cases:
    result = redact_sensitive(test)
    assert "AIza" not in result
    assert "sk-" not in result
    assert "secret123" not in result
    print(f"✅ {result}")
```

## What Gets Logged (Safe)

After redaction, your logs will look like:

```
Error: 404 Not Found for url 'https://api.example.com/v1/chat?key=***REDACTED***'
Tool called: get_weather with {'city': 'Mumbai'}
LLM call completed in 1234ms, tokens: 150
```

## Adding New Patterns

If you find a pattern that's not being redacted, add it to `SENSITIVE_PATTERNS`:

```python
SENSITIVE_PATTERNS = [
    # ... existing patterns ...
    (re.compile(r'your-new-pattern'), '***REDACTED***'),
]
```

## Remember

**When in doubt, redact it out!**

It's better to over-redact than to leak a single API key.
