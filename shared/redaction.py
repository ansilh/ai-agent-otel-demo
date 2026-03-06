"""
=============================================================================
Sensitive Data Redaction Utilities
=============================================================================

This module provides utilities to redact sensitive information from:
- Log messages
- Trace span attributes
- Metric labels
- Error messages

CRITICAL: Always use these utilities when logging or recording any data
that might contain sensitive information like API keys, tokens, passwords,
or PII (Personally Identifiable Information).

Patterns detected and redacted:
- API keys (various formats)
- Bearer tokens
- Authorization headers
- URLs with embedded credentials
- Common secret patterns
=============================================================================
"""

import re
from typing import Any, Dict, Optional
from functools import wraps


# -----------------------------------------------------------------------------
# Sensitive Data Patterns
# -----------------------------------------------------------------------------
# These regex patterns match common sensitive data formats

SENSITIVE_PATTERNS = [
    # API Keys - various formats
    (r'(?i)(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?', r'\1=***REDACTED***'),
    (r'(?i)(key)["\s:=]+["\']?([a-zA-Z0-9_\-]{32,})["\']?', r'\1=***REDACTED***'),
    
    # Bearer tokens
    (r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]+)', r'\1***REDACTED***'),
    
    # Authorization headers
    (r'(?i)(authorization)["\s:=]+["\']?([^"\'}\s]+)["\']?', r'\1=***REDACTED***'),
    
    # Google API keys (AIza...)
    (r'AIza[a-zA-Z0-9_\-]{35}', '***GOOGLE_API_KEY_REDACTED***'),
    
    # Azure keys
    (r'(?i)(azure[_-]?(?:openai)?[_-]?(?:api)?[_-]?key)["\s:=]+["\']?([a-zA-Z0-9]{32,})["\']?', r'\1=***REDACTED***'),
    
    # OpenAI keys (sk-...)
    (r'sk-[a-zA-Z0-9]{48,}', '***OPENAI_KEY_REDACTED***'),
    
    # Generic secrets
    (r'(?i)(secret|password|passwd|pwd|token|credential)["\s:=]+["\']?([^"\'}\s]{8,})["\']?', r'\1=***REDACTED***'),
    
    # URLs with credentials
    (r'(https?://)[^:]+:[^@]+@', r'\1***:***@'),
    
    # URL query params with keys
    (r'(?i)(\?|&)(api[_-]?key|key|token|secret)=([^&\s]+)', r'\1\2=***REDACTED***'),
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [(re.compile(pattern), replacement) for pattern, replacement in SENSITIVE_PATTERNS]


# -----------------------------------------------------------------------------
# Redaction Functions
# -----------------------------------------------------------------------------

def redact_sensitive_data(text: str) -> str:
    """
    Redact sensitive data from a string.
    
    Args:
        text: The input string that may contain sensitive data
        
    Returns:
        The string with sensitive data replaced with ***REDACTED***
        
    Example:
        >>> redact_sensitive_data("API key is AIzaSyABC123...")
        'API key is ***GOOGLE_API_KEY_REDACTED***'
    """
    if not text or not isinstance(text, str):
        return text
    
    result = text
    for pattern, replacement in COMPILED_PATTERNS:
        result = pattern.sub(replacement, result)
    
    return result


def redact_dict(data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Redact sensitive data from dictionary values.
    
    Args:
        data: Dictionary that may contain sensitive data in values
        deep: If True, recursively redact nested dictionaries
        
    Returns:
        Dictionary with sensitive data redacted
    """
    if not data or not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        # Check if key itself suggests sensitive data
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in ['key', 'secret', 'token', 'password', 'credential', 'auth']):
            result[key] = '***REDACTED***'
        elif isinstance(value, str):
            result[key] = redact_sensitive_data(value)
        elif isinstance(value, dict) and deep:
            result[key] = redact_dict(value, deep=True)
        elif isinstance(value, list):
            result[key] = [redact_sensitive_data(str(v)) if isinstance(v, str) else v for v in value]
        else:
            result[key] = value
    
    return result


def safe_str(obj: Any, max_length: int = 500) -> str:
    """
    Convert object to string with sensitive data redacted and length limited.
    
    Args:
        obj: Any object to convert to string
        max_length: Maximum length of output string
        
    Returns:
        Safe string representation
    """
    try:
        text = str(obj)
        redacted = redact_sensitive_data(text)
        if len(redacted) > max_length:
            return redacted[:max_length] + "...[TRUNCATED]"
        return redacted
    except Exception:
        return "[UNABLE TO CONVERT TO STRING]"


# -----------------------------------------------------------------------------
# Decorator for Safe Logging
# -----------------------------------------------------------------------------

def safe_log_args(func):
    """
    Decorator that redacts sensitive data from function arguments before logging.
    
    Use this on functions that log their arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redact kwargs
        safe_kwargs = redact_dict(dict(kwargs))
        # Redact args (convert to strings and redact)
        safe_args = tuple(safe_str(arg) for arg in args)
        return func(*safe_args, **safe_kwargs)
    return wrapper


# -----------------------------------------------------------------------------
# Safe Span Attribute Setter
# -----------------------------------------------------------------------------

class SafeSpanAttributes:
    """
    Helper class to safely set span attributes with automatic redaction.
    
    Usage:
        with tracer.start_as_current_span("my_span") as span:
            safe_attrs = SafeSpanAttributes(span)
            safe_attrs.set("user.input", potentially_sensitive_data)
    """
    
    # Attributes that should NEVER be recorded (always redact completely)
    FORBIDDEN_ATTRIBUTES = {
        'api_key', 'apikey', 'api.key',
        'secret', 'password', 'passwd', 'pwd',
        'token', 'auth_token', 'access_token', 'refresh_token',
        'credential', 'credentials',
        'authorization', 'auth',
        'private_key', 'private.key',
    }
    
    def __init__(self, span):
        self.span = span
    
    def set(self, key: str, value: Any) -> None:
        """
        Safely set a span attribute with automatic redaction.
        
        Args:
            key: Attribute name
            value: Attribute value (will be redacted if sensitive)
        """
        # Check if key is forbidden
        key_lower = key.lower().replace('.', '_').replace('-', '_')
        if any(forbidden in key_lower for forbidden in self.FORBIDDEN_ATTRIBUTES):
            # Don't record forbidden attributes at all
            return
        
        # Redact the value
        if isinstance(value, str):
            safe_value = redact_sensitive_data(value)
        elif isinstance(value, dict):
            safe_value = str(redact_dict(value))
        else:
            safe_value = safe_str(value)
        
        self.span.set_attribute(key, safe_value)


# -----------------------------------------------------------------------------
# Environment Variable Safety
# -----------------------------------------------------------------------------

def get_safe_env_summary() -> Dict[str, str]:
    """
    Get a summary of relevant environment variables with values redacted.
    Useful for debugging without exposing secrets.
    
    Returns:
        Dictionary with env var names and redacted values
    """
    import os
    
    relevant_vars = [
        'OTEL_EXPORTER_OTLP_ENDPOINT',
        'OTEL_SERVICE_NAME',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT',
        'AZURE_OPENAI_API_VERSION',
        'GOOGLE_API_KEY',
        'AZURE_OPENAI_API_KEY',
        'OPENAI_API_KEY',
    ]
    
    result = {}
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            if 'KEY' in var.upper() or 'SECRET' in var.upper() or 'TOKEN' in var.upper():
                result[var] = '***SET***'
            else:
                result[var] = value
        else:
            result[var] = '(not set)'
    
    return result


# =============================================================================
# IMPORTANT SECURITY NOTES
# =============================================================================
#
# 1. NEVER log raw API responses without redaction
# 2. NEVER include API keys in span attributes
# 3. NEVER include full URLs with query parameters containing keys
# 4. ALWAYS use safe_str() when converting unknown objects to strings
# 5. ALWAYS use SafeSpanAttributes when setting span attributes
# 6. ALWAYS use redact_dict() when logging dictionaries
#
# If you find a pattern that's not being redacted, ADD IT to SENSITIVE_PATTERNS
# =============================================================================
