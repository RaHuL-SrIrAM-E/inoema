import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "app.py")


def test_dashboard_loads_without_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception


def test_dashboard_runs_full_demo_on_button_click():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.button[0].click().run(timeout=120)
    assert not at.exception

    # All five report sections should have rendered after a completed run.
    subheader_text = [h.value for h in at.subheader]
    for expected in (
        "Agent Control",
        "Simulation Results",
        "Execution — Mock PayPal Gateway",
        "Independent Verification",
        "Agent Execution Timeline",
    ):
        assert expected in subheader_text


def test_developer_integration_renders_before_a_run():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    subheader_text = [h.value for h in at.subheader]
    assert "Developer Integration" in subheader_text
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "REST API" in markdown_text
    assert "MCP Tool Catalogue" in markdown_text
    assert "Architecture" in markdown_text


def test_developer_integration_mcp_trace_button_after_run():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    at.button[0].click().run(timeout=120)  # Run INOEMA Simulation
    assert not at.exception

    mcp_button = next(b for b in at.button if b.label == "Run example MCP interaction")
    mcp_button.click().run(timeout=30)
    assert not at.exception

    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Agent discovers available INOEMA tools" in markdown_text
