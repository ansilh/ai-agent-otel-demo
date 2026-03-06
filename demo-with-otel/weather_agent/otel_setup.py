"""
=============================================================================
OpenTelemetry Setup Module
=============================================================================

This module initializes OpenTelemetry for the ADK agent.
It configures traces, metrics, and logging to be sent to SigNoz.

The setup is done ONCE at module import time, so it's ready before
the ADK web server starts handling requests.

Configuration via environment variables:
- OTEL_EXPORTER_OTLP_ENDPOINT: SigNoz endpoint (default: http://localhost:4317)
- OTEL_SERVICE_NAME: Service name in SigNoz (default: weather-agent-otel)
=============================================================================
"""

import os
import re
import logging

# -----------------------------------------------------------------------------
# Sensitive Data Redaction (CRITICAL FOR SECURITY)
# -----------------------------------------------------------------------------
# NEVER log API keys, tokens, or credentials in traces, metrics, or logs

SENSITIVE_PATTERNS = [
    (re.compile(r'AIza[a-zA-Z0-9_\-]{35}'), '***GOOGLE_API_KEY***'),
    (re.compile(r'sk-[a-zA-Z0-9]{48,}'), '***OPENAI_KEY***'),
    (re.compile(r'(?i)(api[_-]?key|key|token|secret|password|credential|auth)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?'), r'\1=***REDACTED***'),
    (re.compile(r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]+)'), r'\1***REDACTED***'),
    (re.compile(r'(https?://)[^:]+:[^@]+@'), r'\1***:***@'),
    (re.compile(r'(?i)(\?|&)(api[_-]?key|key|token)=([^&\s]+)'), r'\1\2=***REDACTED***'),
]

def redact_sensitive(text: str) -> str:
    """Redact sensitive data from text. ALWAYS use this before logging."""
    if not text or not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result

def safe_str(obj, max_length: int = 500) -> str:
    """Convert to string with redaction and length limit."""
    try:
        text = str(obj)
        redacted = redact_sensitive(text)
        return redacted[:max_length] + '...' if len(redacted) > max_length else redacted
    except Exception:
        return '[UNABLE TO CONVERT]'

# -----------------------------------------------------------------------------
# OpenTelemetry Imports
# -----------------------------------------------------------------------------

# Core APIs - these are the interfaces we use to create traces and metrics
from opentelemetry import trace, metrics

# SDK components - these implement the actual tracing/metrics logic
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Exporters - these send data to SigNoz via OTLP protocol
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Processors - these batch and process data before export
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Status codes for marking spans as success/error
from opentelemetry.trace import Status, StatusCode


def setup_opentelemetry():
    """
    Initialize OpenTelemetry with traces and metrics.
    
    This function sets up the "plumbing" that sends observability data
    to SigNoz. It should be called once at application startup.
    
    Returns:
        tuple: (tracer, meter) for creating spans and recording metrics
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Create Resource
    # -------------------------------------------------------------------------
    # Resource identifies this service in SigNoz
    # All traces/metrics will be tagged with this info
    service_name = os.environ.get("OTEL_SERVICE_NAME", "weather-agent-otel")
    
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "demo",
    })
    
    # -------------------------------------------------------------------------
    # Step 2: Get OTLP Endpoint
    # -------------------------------------------------------------------------
    # SigNoz listens on port 4317 for OTLP/gRPC
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # -------------------------------------------------------------------------
    # Step 3: Set up Tracing
    # -------------------------------------------------------------------------
    # Traces show request flow through the system
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Create exporter to send traces to SigNoz
    trace_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True,  # Use insecure for local development
    )
    
    # BatchSpanProcessor batches spans for efficient export
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    
    # Register globally so all code can access it
    trace.set_tracer_provider(tracer_provider)
    
    # Get tracer instance
    tracer = trace.get_tracer("weather-agent", "1.0.0")
    
    # -------------------------------------------------------------------------
    # Step 4: Set up Metrics
    # -------------------------------------------------------------------------
    # Metrics are numerical measurements over time
    
    # Create metric exporter
    metric_exporter = OTLPMetricExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )
    
    # Export metrics every 10 seconds (use 60s in production)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=10000,
    )
    
    # Create and register meter provider
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )
    metrics.set_meter_provider(meter_provider)
    
    # Get meter instance
    meter = metrics.get_meter("weather-agent", "1.0.0")
    
    # -------------------------------------------------------------------------
    # Step 5: Configure Logging
    # -------------------------------------------------------------------------
    # Set up logging to include trace context
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )
    
    return tracer, meter


def create_metrics(meter):
    """
    Create metric instruments for tracking agent performance.
    
    Args:
        meter: OpenTelemetry meter instance
        
    Returns:
        dict: Dictionary of metric instruments
    """
    return {
        # Token counters
        "input_tokens": meter.create_counter(
            name="agent.tokens.input",
            description="Total input tokens used",
            unit="tokens",
        ),
        "output_tokens": meter.create_counter(
            name="agent.tokens.output",
            description="Total output tokens generated",
            unit="tokens",
        ),
        
        # Request latency histogram
        "request_latency": meter.create_histogram(
            name="agent.request.latency",
            description="Request latency in milliseconds",
            unit="ms",
        ),
        
        # Tool call counter
        "tool_calls": meter.create_counter(
            name="agent.tool.calls",
            description="Number of tool calls made",
            unit="calls",
        ),
        
        # Error counter
        "errors": meter.create_counter(
            name="agent.errors",
            description="Number of errors encountered",
            unit="errors",
        ),
    }


# -----------------------------------------------------------------------------
# Initialize OTEL at module load time
# -----------------------------------------------------------------------------
# This ensures OTEL is ready before ADK web starts

tracer, meter = setup_opentelemetry()
agent_metrics = create_metrics(meter)
logger = logging.getLogger("weather-agent")

# Export for use in agent.py
__all__ = ['tracer', 'meter', 'agent_metrics', 'logger', 'Status', 'StatusCode', 'redact_sensitive', 'safe_str']
