import pytest
from pydantic import ValidationError

from groundedai.core.config import Settings
from groundedai.core.errors import RateLimited
from groundedai.observability.metrics import MetricsRegistry, percentile, summarize
from groundedai.security.rate_limit import Limit, TokenBucketLimiter


def test_production_requires_a_strong_api_key() -> None:
    with pytest.raises(ValidationError, match="GROUNDEDAI_API_KEY"):
        Settings(_env_file=None, env="production")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, env="production", api_key="too-short")
    settings = Settings(_env_file=None, env="production", api_key="x" * 32)
    assert settings.auth_enabled


def test_production_forbids_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="Wildcard"):
        Settings(_env_file=None, env="production", api_key="x" * 32, cors_origins=["*"])


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_cors_origins_accept_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDEDAI_CORS_ORIGINS", "https://a.example, https://b.example")
    assert Settings(_env_file=None).cors_origins == ["https://a.example", "https://b.example"]


def test_default_database_is_sqlite_in_data_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.resolved_database_url == f"sqlite:///{(tmp_path / 'groundedai.db').as_posix()}"
    assert not settings.auth_enabled


def test_token_bucket_refills_over_time() -> None:
    now = [0.0]
    limiter = TokenBucketLimiter(clock=lambda: now[0])
    limit = Limit(capacity=2, period_s=60)
    limiter.check("query", "ip:1", limit)
    limiter.check("query", "ip:1", limit)
    with pytest.raises(RateLimited) as error:
        limiter.check("query", "ip:1", limit)
    assert error.value.retry_after_s == 30
    limiter.check("query", "ip:2", limit)  # other clients are independent
    now[0] = 30.0
    limiter.check("query", "ip:1", limit)


def test_percentile_and_summary() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([], 50) is None
    assert summarize([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    assert summarize([1.0, 3.0])["mean"] == 2.0


def test_prometheus_exposition() -> None:
    registry = MetricsRegistry()
    registry.inc("http_requests", route="/api/v1/query", status_class="2xx")
    registry.observe("query_latency", 12.5)
    text = registry.prometheus()
    assert "# TYPE groundedai_http_requests_total counter" in text
    assert 'groundedai_http_requests_total{route="/api/v1/query",status_class="2xx"} 1' in text
    assert 'groundedai_query_latency_ms{quantile="0.5"} 12.500' in text
    assert registry.counter_total("http_requests", status_class="2xx") == 1


def test_loopback_urls_bypass_environment_proxies() -> None:
    # Regression: with HTTP_PROXY set, local Ollama requests went to the proxy (HTTP 503).
    from groundedai.core.net import http_client_options, is_loopback_url

    assert is_loopback_url("http://localhost:11434")
    assert is_loopback_url("http://127.0.0.1:11434")
    assert is_loopback_url("http://[::1]:11434")
    assert not is_loopback_url("http://ollama:11434")
    assert not is_loopback_url("https://models.example.com")
    assert http_client_options("http://127.0.0.1:11434", 5)["trust_env"] is False
    assert http_client_options("http://ollama:11434", 5)["trust_env"] is True
