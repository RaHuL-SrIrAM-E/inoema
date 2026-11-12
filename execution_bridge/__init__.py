"""Policy execution bridge: persisted optimization result -> validated, versioned execution plan -> deterministic executor.

Pure and offline: nothing here calls Gemini, MCP or PayPal. The only PayPal touch point is `paypal_gate`, which refuses
simulated and completed payments and otherwise only invokes an *injected* callable (the existing, verified ResumeCapture).
"""
