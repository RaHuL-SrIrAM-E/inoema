"""OAuth2 client-credentials token handling. Tokens live only in memory and are never logged or persisted."""
from __future__ import annotations

import time
from typing import Any, Callable

from paypal.models import PayPalApiError, PayPalConfig


class TokenProvider:
    SKEW_S = 60.0

    def __init__(self, config: PayPalConfig, send: Callable[..., dict], clock: Callable[[], float] = time.time):
        self._cfg, self._send, self._clock = config, send, clock
        self._token: str | None = None
        self._expires_at = 0.0
        self.fetch_count = 0

    def get_token(self) -> str:
        if self._token is None or self._clock() >= self._expires_at - self.SKEW_S:
            self._fetch()
        assert self._token is not None
        return self._token

    def invalidate(self) -> None:
        self._token, self._expires_at = None, 0.0

    def _fetch(self) -> None:
        body = self._send("POST", "/v1/oauth2/token", operation="OAUTH_TOKEN", data={"grant_type": "client_credentials"},
                          headers={"Accept": "application/json"}, basic_auth=(self._cfg.client_id, self._cfg.client_secret))
        tok = body.get("access_token")
        if not tok:
            raise PayPalApiError(200, "MALFORMED_TOKEN_RESPONSE", "token response had no access_token", operation="OAUTH_TOKEN")
        self._token = tok
        self._expires_at = self._clock() + float(body.get("expires_in", 300))
        self.fetch_count += 1
