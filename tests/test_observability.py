from __future__ import annotations

from biomcp.observability import (
    record_auth_event,
    record_cache_event,
    record_http_request,
    record_tool_call,
    record_upstream_error,
    record_upstream_request,
    render_prometheus_metrics,
)


def test_render_prometheus_metrics_includes_core_series():
    record_http_request("/mcp", "POST", 200, "oauth")
    record_tool_call("multi_omics_gene_report", "success", 1.23)
    record_cache_event("multi_omics", "hit")
    record_upstream_request("api.ncbi.nlm.nih.gov", 200, 0.42)
    record_upstream_error("clinicaltrials.gov", "ConnectTimeout")
    record_auth_event("token_accepted")

    output = render_prometheus_metrics()

    assert "biomcp_http_requests_total" in output
    assert 'path="/mcp"' in output
    assert "biomcp_tool_calls_total" in output
    assert 'tool="multi_omics_gene_report"' in output
    assert "biomcp_tool_latency_seconds" in output
    assert 'quantile="0.50"' in output
    assert "biomcp_cache_events_total" in output
    assert "biomcp_upstream_http_requests_total" in output
    assert "biomcp_upstream_errors_total" in output
    assert "biomcp_auth_events_total" in output
