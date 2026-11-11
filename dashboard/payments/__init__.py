"""INOEMA Payments workspace for the Streamlit dashboard.

Layering: `data`/`status`/`runner`/`sandbox_service` are Streamlit-free (testable, reusable); `ui` + `page_*` only render.
No simulation, agent or PayPal business logic is implemented here: the UI calls the existing backend modules.
"""
