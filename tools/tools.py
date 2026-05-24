"""Tools the assistant can call.

Three tools, deliberately minimal:
- get_datetime: now() in the user's preferred TZ (default UTC)
- calculator: evaluate an arithmetic expression with sympy (no eval())
- web_search: DuckDuckGo HTML scrape, no API key required, top 3 hits

The dispatcher returns a structured result that the assistant feeds back into
the model as an "observation".
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict


def get_datetime(timezone: str = "UTC") -> Dict[str, Any]:
    now = _dt.datetime.now(_dt.timezone.utc)
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %B %d %Y, %H:%M %Z"),
        "timezone": timezone,
    }


def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluate an arithmetic expression."""
    try:
        from sympy import sympify

        result = sympify(expression)
        return {"expression": expression, "result": str(result)}
    except Exception as e:  # noqa: BLE001
        return {"expression": expression, "error": f"could not evaluate: {e}"}


def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Top-k DuckDuckGo results. Returns title + snippet + url."""
    try:
        try:
            from ddgs import DDGS  # new package name (post-rename)
        except ImportError:
            from duckduckgo_search import DDGS  # older versions

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return {
            "query": query,
            "results": [
                {
                    "title": h.get("title", ""),
                    "snippet": h.get("body", ""),
                    "url": h.get("href", ""),
                }
                for h in hits
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {"query": query, "error": f"search failed: {e}", "results": []}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_datetime": get_datetime,
    "calculator": calculator,
    "web_search": web_search,
}


def tool_specs() -> list[dict]:
    """JSON-schema specs, used by both the Anthropic tool API and the OSS prompt."""
    return [
        {
            "name": "get_datetime",
            "description": "Returns the current date and time. Use when the user asks about today, now, or scheduling.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name; defaults to UTC.",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "calculator",
            "description": "Evaluates an arithmetic expression. Use for math the user asks you to compute exactly.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '2*(3+4)/7'.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the public web for current information. Use when the user asks about anything that may have changed recently or that you do not know.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (1-5).",
                    },
                },
                "required": ["query"],
            },
        },
    ]


def run_tool(name: str, **kwargs) -> Dict[str, Any]:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**kwargs)
    except TypeError as e:
        return {"error": f"bad arguments to {name}: {e}"}
