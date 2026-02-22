from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "model_b_request_count",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "model_b_request_latency_seconds",
    "Latency of prediction requests"
)

ERROR_COUNT = Counter(
    "model_b_error_count",
    "Total number of prediction errors"
)