"""Helpers shared by the speculative-decoding driver."""

import httpx

TOKEN_ID_PREFIX = "token_id:"


def parse_token_id(s: str) -> int:
    """Servers return tokens as 'token_id:<int>' when return_tokens_as_token_ids=True."""
    if not s.startswith(TOKEN_ID_PREFIX):
        raise ValueError(f"Expected '{TOKEN_ID_PREFIX}<int>', got {s!r}. "
                         "Does your server support "
                         "return_tokens_as_token_ids?")
    return int(s[len(TOKEN_ID_PREFIX):])


async def fetch_model_id(client: httpx.AsyncClient, url: str) -> str:
    """GET /v1/models on the server and return the served model id."""
    try:
        r = await client.get(f"{url}/v1/models")
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"<could not fetch: {e}>"
    entries = r.json().get("data", [])
    return ", ".join(e.get("id", "?") for e in entries) or "<none>"
