r"""
Memory-Augmented Multi-Agent Travel Assistant  (single-file edition)
=====================================================================

One file, five commands:

    streamlit run travel_assistant.py          -> the web app
    python travel_assistant.py check           -> verify keys, database and all 3 MCP servers
    python travel_assistant.py chat            -> chat in the terminal (same graph, same memory)
    python travel_assistant.py selftest        -> offline tests (no API keys needed)
    python travel_assistant.py tools-server    -> our MCP server: weather + route flights (the app starts it for you)

What makes it different from a plain "LLM + APIs" travel bot
-----------------------------------------------------------
1. Travel DNA      - a saved, editable traveller profile (pace, food, stay style, accessibility...)
                     stored in PostgreSQL and applied automatically to every new trip.
2. Memory you control - things you *say* about yourself become *suggestions*; they are never used
                     as facts until you accept them. Trip-only wishes stay in that trip only.
3. "Why this fits you" - every recommendation names the DNA fields and the sources behind it,
                     and the validator rejects references to preferences or sources that don't exist.
4. Reality Check   - a rule-based checker scores the plan (budget, pace, weather, grounding,
                     preference fit), proposes computed alternatives, and triggers at most ONE
                     automatic revision.
5. Rain Plan B     - outdoor activities on days the forecast says are risky must carry an indoor
                     backup, or the Reality Check sends the plan back.
6. Honest data     - every external fact keeps its provider, URL, retrieval time and the date it
                     applies to; flights/weather/search report ok / empty / unavailable / error
                     instead of inventing data. Fares and bookings are never claimed.

Architecture (LangGraph):
    intake -> [flights | weather | research]  (parallel, via MCP) -> planner -> reality_check
           \-> finalize (clarifying question)              planner <- (max 1 revision) -/
                                                                       reality_check -> finalize
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import operator
import os
import re
import shlex
import sys
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

try:  # Python 3.11+ has this built in; on 3.10 it comes from the `exceptiongroup` package
    BaseExceptionGroup
except NameError:  # pragma: no cover
    from exceptiongroup import BaseExceptionGroup

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
logging.basicConfig(stream=sys.stderr, level=os.getenv("LOG_LEVEL", "WARNING"))
log = logging.getLogger("travel")

# =============================================================================
# 1. SETTINGS
# =============================================================================
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TRIP_DAYS = 14
MAX_REVISIONS = 1
PACE_CAPS = {"relaxed": 3, "balanced": 4, "packed": 6}  # activities per day (meals/rest/travel excluded)
PACE_ORDER = ["relaxed", "balanced", "packed"]

# The Travel DNA fields. "list" fields hold several values.
PROFILE_FIELDS: dict[str, str] = {
    "travel_style": "list",          # budget-conscious, comfort, adventure, culture, food, nature, nightlife, relaxed
    "pace": "str",                   # relaxed | balanced | packed
    "accommodation": "str",          # hostel, budget hotel, mid-range hotel, boutique, luxury, homestay
    "food": "str",                   # vegetarian, vegan, halal, jain, no restrictions...
    "accessibility": "str",          # e.g. wheelchair user, limited walking
    "transport": "str",              # public transport, taxis, walking, self-drive
    "favourite_activities": "list",  # museums, food markets, hiking...
    "budget_range": "str",           # free text, e.g. "₹8k-12k per day"
    "avoid": "list",                 # things to keep out of plans
}


@dataclass
class Settings:
    groq_key: str
    groq_model: str
    tavily_key: str
    aviation_key: str
    aviation_plan: str
    owm_key: str
    database_url: str
    contingency: float
    mcp_timeout: float

    @classmethod
    def load(cls) -> "Settings":
        def g(name: str, default: str = "") -> str:
            return (os.getenv(name) or default).strip()

        plan = g("AVIATIONSTACK_PLAN", "free").lower()
        return cls(
            groq_key=g("GROQ_API_KEY"),
            groq_model=g("GROQ_MODEL", DEFAULT_MODEL),
            tavily_key=g("TAVILY_API_KEY"),
            aviation_key=g("AVIATIONSTACK_API_KEY"),
            aviation_plan=plan if plan in ("free", "basic") else "free",
            owm_key=g("OPENWEATHER_API_KEY"),
            database_url=g("DATABASE_URL"),
            contingency=float(g("BUDGET_CONTINGENCY", "0.10")),
            mcp_timeout=float(g("MCP_TIMEOUT_SECONDS", "60")),
        )

    def problems(self) -> list[str]:
        """Human-readable setup problems. Only Groq is strictly required."""
        out = []
        if not self.groq_key:
            out.append("GROQ_API_KEY is missing - the planner cannot run without it.")
        if not self.tavily_key:
            out.append("TAVILY_API_KEY is missing - stays/activities search will be unavailable.")
        if not self.owm_key:
            out.append("OPENWEATHER_API_KEY is missing - weather will be unavailable.")
        if not self.aviation_key:
            out.append("AVIATIONSTACK_API_KEY is missing - flight lookups will be unavailable.")
        if not self.database_url:
            out.append("DATABASE_URL is missing - memory will be temporary (lost on restart).")
        return out


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def short_error(exc: BaseException) -> str:
    """Unwrap exception groups (anyio/MCP raise these) and hide API keys."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    msg = re.sub(r"(access_key|appid|apiKey|tavilyApiKey|api_key)=[^&\s]+", r"\1=***", msg)
    return msg[:300]


# =============================================================================
# 2. DATA SCHEMAS  (pydantic validates everything the LLM produces)
# =============================================================================
class TripRequest(BaseModel):
    origin_city: Optional[str] = None
    origin_iata: Optional[str] = None
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    destination_iata: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    travellers: Optional[int] = None
    budget_amount: Optional[float] = None
    budget_currency: Optional[str] = None
    budget_includes_flights: Optional[bool] = None
    interests: list[str] = Field(default_factory=list)
    trip_preferences: dict[str, Any] = Field(default_factory=dict, description="Preferences for THIS trip only")
    flight_numbers: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class PreferenceStatement(BaseModel):
    field: str
    value: Any
    quote: str = ""


class IntakeResult(BaseModel):
    trip: TripRequest
    lasting_preferences: list[PreferenceStatement] = Field(default_factory=list)
    clarification_question: Optional[str] = None


ITEM_KINDS = ("outdoor", "indoor", "meal", "travel", "rest")


class Item(BaseModel):
    start: str = "09:00"
    end: str = "10:00"
    title: str
    kind: str = "indoor"
    est_cost: float = 0.0
    cost_basis: str = "estimate"  # "estimate" (LLM) or "source" (stated by a cited source)
    why: str = ""
    preference_fields: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rain_alternative: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        v = (v or "").lower().strip()
        return v if v in ITEM_KINDS else "indoor"

    @field_validator("cost_basis")
    @classmethod
    def _basis(cls, v: str) -> str:
        return "source" if (v or "").lower().strip() == "source" else "estimate"

    @field_validator("est_cost", mode="before")
    @classmethod
    def _cost(cls, v: Any) -> float:
        try:
            return max(0.0, float(v or 0))
        except (TypeError, ValueError):
            return 0.0


class Day(BaseModel):
    date: str
    theme: str = ""
    items: list[Item] = Field(default_factory=list)


class Stay(BaseModel):
    name: str
    area: str = ""
    why: str = ""
    price_note: str = ""
    est_total_cost: float = 0.0  # whole stay, all travellers (the FIRST stay is the one counted in the budget)
    preference_fields: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("est_total_cost", mode="before")
    @classmethod
    def _cost(cls, v: Any) -> float:
        try:
            return max(0.0, float(v or 0))
        except (TypeError, ValueError):
            return 0.0


class Itinerary(BaseModel):
    summary: str = ""
    personalization_note: str = ""
    currency: str = "INR"
    stays: list[Stay] = Field(default_factory=list)
    days: list[Day] = Field(default_factory=list)
    estimated_airfare: Optional[float] = None
    assumptions: list[str] = Field(default_factory=list)


# =============================================================================
# 3. LLM (Groq) - JSON mode + pydantic validation, max 2 attempts
# =============================================================================
class LLMError(RuntimeError):
    pass


def rate_limit_wait(exc: BaseException) -> Optional[float]:
    """Seconds to wait if `exc` is a 429 rate-limit error, else None."""
    if getattr(exc, "status_code", None) != 429:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
        return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 10.0


class GroqLLM:
    """Callable (system, user) -> str. Kept tiny so tests can swap in a fake.

    Groq's free tier limits tokens per minute; on a 429 we wait (up to MAX_WAIT seconds,
    at most twice) instead of failing. `on_wait` lets the UI show what is happening."""

    MAX_WAIT = 45

    def __init__(self, settings: Settings, client: Any = None, sleep: Callable[[float], None] | None = None):
        if client is None:
            from groq import Groq

            client = Groq(api_key=settings.groq_key, timeout=90, max_retries=0)
        self.client = client
        self.model = settings.groq_model
        self.sleep = sleep or __import__("time").sleep
        self.on_wait: Callable[[str], None] | None = None

    def __call__(self, system: str, user: str) -> str:
        extra = {"reasoning_effort": "low"} if self.model.startswith("openai/gpt-oss") else {}
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_completion_tokens=6000,
                    **extra,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # network, rate limit, bad model name...
                wait = rate_limit_wait(exc)
                if wait is None:
                    raise LLMError(f"Groq call failed ({self.model}): {short_error(exc)}") from exc
                if attempt == 2 or wait > self.MAX_WAIT:
                    raise LLMError(
                        f"Groq free-tier rate limit reached for {self.model} (needs a {wait:.0f}s wait). "
                        "Wait a minute and try again, or set a different GROQ_MODEL in .env."
                    ) from exc
                if self.on_wait:
                    self.on_wait(f"⏳ Groq free-tier limit reached - waiting {wait:.0f}s, then retrying automatically…")
                self.sleep(wait + 1)
        raise LLMError("unreachable")


def extract_json(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return raw[start : end + 1]


def ask_json(llm: Callable[[str, str], str], model_cls: type[BaseModel], system: str, user: str) -> BaseModel:
    schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
    system_full = f"{system}\n\nReply with ONLY one JSON object that matches this JSON Schema:\n{schema}"
    error = ""
    for attempt in range(2):
        prompt = user if attempt == 0 else f"{user}\n\nYour previous reply was invalid ({error}). Reply with corrected JSON only."
        raw = llm(system_full, prompt)
        try:
            return model_cls.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:400]
    raise LLMError(f"The model returned invalid JSON twice: {error}")


# =============================================================================
# 4. MCP CLIENT  (official `mcp` SDK; one session per server per agent call)
# =============================================================================
class MCPUnavailable(RuntimeError):
    pass


def _server_target(name: str, s: Settings):
    env = dict(os.environ)
    if name == "travel-tools":
        from mcp import StdioServerParameters

        return "stdio", StdioServerParameters(
            command=sys.executable, args=[str(Path(__file__).resolve()), "tools-server"], env=env, cwd=str(HERE)
        )
    if name == "aviationstack":
        from mcp import StdioServerParameters

        posix = os.name != "nt"
        command = os.getenv("AVIATIONSTACK_MCP_COMMAND", "uvx")
        args = shlex.split(os.getenv("AVIATIONSTACK_MCP_ARGS", "--python 3.13 aviationstack-mcp==1.8.1"), posix=posix)
        env["AVIATION_STACK_API_KEY"] = s.aviation_key
        return "stdio", StdioServerParameters(command=command, args=args, env=env)
    if name == "tavily":
        url = os.getenv("TAVILY_MCP_URL", "https://mcp.tavily.com/mcp/")
        return "http", (url, {"Authorization": f"Bearer {s.tavily_key}"})  # key in a header, never in the URL
    raise ValueError(f"unknown MCP server {name}")


@asynccontextmanager
async def open_mcp(name: str, s: Settings):
    from mcp import ClientSession

    kind, target = _server_target(name, s)
    if kind == "stdio":
        from mcp.client.stdio import stdio_client

        async with stdio_client(target) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        from mcp.client.streamable_http import streamablehttp_client

        url, headers = target
        async with streamablehttp_client(url, headers=headers, timeout=30) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def _mcp_calls(name: str, calls: list[tuple[list[str], dict]], s: Settings) -> list[dict]:
    """Discover tools, then call each (candidate_names, args), with an overall timeout."""
    return await asyncio.wait_for(_mcp_calls_inner(name, calls, s), timeout=s.mcp_timeout)


async def _mcp_calls_inner(name: str, calls: list[tuple[list[str], dict]], s: Settings) -> list[dict]:
    async with open_mcp(name, s) as session:
        available = {t.name for t in (await session.list_tools()).tools}
        out = []
        for candidates, args in calls:
            tool = next((c for c in candidates if c in available), None)
            if tool is None:
                out.append({"status": "unavailable", "error": f"tool '{candidates[0]}' not exposed by the {name} server"})
                continue
            res = await session.call_tool(tool, args)
            text = "\n".join(getattr(c, "text", "") for c in res.content if getattr(c, "type", "") == "text")
            out.append(
                {
                    "status": "error" if res.isError else "ok",
                    "text": text,
                    "structured": res.structuredContent,
                    "error": text[:300] if res.isError else None,
                }
            )
        return out


def call_mcp(name: str, calls: list[tuple[list[str], dict]], s: Settings) -> list[dict]:
    """Sync wrapper used by graph nodes. Any connection problem -> MCPUnavailable."""
    try:
        return asyncio.run(_mcp_calls(name, calls, s))
    except Exception as exc:
        raise MCPUnavailable(f"{name} MCP server unavailable: {short_error(exc)}") from exc


async def _list_tools(name: str, s: Settings) -> list[str]:
    async def inner() -> list[str]:
        async with open_mcp(name, s) as session:
            return sorted(t.name for t in (await session.list_tools()).tools)

    return await asyncio.wait_for(inner(), timeout=s.mcp_timeout)


# =============================================================================
# 5. RESPONSE NORMALISERS  (pure functions -> easy to test)
# =============================================================================
def parse_tavily(text: str, structured: Any = None) -> list[dict]:
    """Tavily MCP returns text blocks ('Title:/URL:/Content:'). Accept JSON too."""
    data = structured if isinstance(structured, dict) else None
    if data is None:
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            data = None
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""), "published_date": r.get("published_date")}
            for r in data["results"]
            if isinstance(r, dict) and str(r.get("url", "")).startswith("http")
        ]
    results: list[dict] = []
    cur: dict = {}
    skip = False
    keys = {"title": "title", "url": "url", "content": "content", "published date": "published_date"}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(Title|URL|Content|Published Date)\s*:\s*(.*)$", line, flags=re.I)
        if m:
            key = keys[m.group(1).lower()]
            if key in ("title", "url") and cur.get(key):
                results.append(cur)
                cur = {}
            cur[key] = m.group(2).strip()
            skip = False
        elif re.match(r"^\s*(ID|Favicon|Raw Content|Request ID|Answer|Detailed Results|Images?)\s*:?", line, flags=re.I):
            skip = True
        elif not skip and "content" in cur and line.strip():
            cur["content"] += " " + line.strip()
    if cur:
        results.append(cur)
    return [r for r in results if str(r.get("url", "")).startswith("http")]


PLAN_RESTRICTED = ("access_restricted", "function_access_restricted", "subscription plan", "upgrade", "basic plan", "not available on your")


def parse_aviation(text: str) -> dict:
    """aviationstack-mcp envelope: {"ok": true, "count": n, "data": [...]} or {"ok": false, "error": "..."}."""
    try:
        env = json.loads(text)
    except (TypeError, ValueError):
        return {"status": "error", "reason": "malformed AviationStack response", "data": []}
    if not isinstance(env, dict) or "ok" not in env:
        return {"status": "error", "reason": "unexpected AviationStack response shape", "data": []}
    if not env["ok"]:
        err = str(env.get("error", "unknown error"))
        if any(k in err.lower() for k in PLAN_RESTRICTED):
            return {"status": "unavailable", "reason": f"Not included in your AviationStack plan ({err[:120]})", "data": []}
        return {"status": "error", "reason": f"AviationStack error: {err[:200]}", "data": []}
    data = env.get("data") or []
    if not isinstance(data, list):
        return {"status": "error", "reason": "AviationStack 'data' is not a list", "data": []}
    return {"status": "ok" if data else "empty", "reason": env.get("message", "") or ("" if data else "no flights found"), "data": data}


# =============================================================================
# 6. OUR WEATHER MCP SERVER  (OpenWeatherMap: geocoding + current + 5-day/3-hour)
# =============================================================================
def _owm_get(path: str, params: dict) -> Any:
    import httpx

    key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENWEATHER_API_KEY is not set")
    base = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org").rstrip("/")
    resp = httpx.get(f"{base}{path}", params={**params, "appid": key}, timeout=10)
    if resp.status_code == 401:
        raise ValueError("OpenWeatherMap rejected the key (401). New keys can take 1-2 hours to activate.")
    resp.raise_for_status()
    return resp.json()


def _geocode(city: str) -> dict:
    data = _owm_get("/geo/1.0/direct", {"q": city, "limit": 1})
    if not isinstance(data, list) or not data:
        raise ValueError(f"City not found by OpenWeatherMap geocoding: {city!r}")
    g = data[0]
    return {"name": g.get("name", city), "country": g.get("country", ""), "lat": g["lat"], "lon": g["lon"]}


def weather_current(city: str) -> dict:
    g = _geocode(city)
    d = _owm_get("/data/2.5/weather", {"lat": g["lat"], "lon": g["lon"], "units": "metric"})
    return {
        "ok": True,
        "location": g,
        "temperature_c": d["main"]["temp"],
        "feels_like_c": d["main"].get("feels_like"),
        "humidity": d["main"].get("humidity"),
        "conditions": (d.get("weather") or [{}])[0].get("description", ""),
        "wind_mps": (d.get("wind") or {}).get("speed"),
        "source": "OpenWeatherMap current weather",
        "retrieved_at": now_iso(),
    }


def weather_forecast(city: str, start_date: str, end_date: str) -> dict:
    start, end = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    if end < start or (end - start).days > 30:
        raise ValueError("invalid date range")
    g = _geocode(city)
    d = _owm_get("/data/2.5/forecast", {"lat": g["lat"], "lon": g["lon"], "units": "metric"})
    tz = dt.timedelta(seconds=int((d.get("city") or {}).get("timezone", 0)))
    buckets: dict[str, dict] = {}
    for slot in d.get("list", []):
        local_day = (dt.datetime.fromtimestamp(slot["dt"], dt.timezone.utc) + tz).date().isoformat()
        b = buckets.setdefault(local_day, {"temps": [], "pops": [], "conds": []})
        b["temps"].append(slot["main"]["temp"])
        b["pops"].append(float(slot.get("pop", 0)))
        b["conds"].append((slot.get("weather") or [{}])[0].get("description", ""))
    days, not_covered = [], []
    cur = start
    while cur <= end:
        key = cur.isoformat()
        b = buckets.get(key)
        if b:
            common = sorted(set(b["conds"]), key=b["conds"].count, reverse=True)[:2]
            days.append(
                {
                    "date": key,
                    "temp_min_c": round(min(b["temps"]), 1),
                    "temp_max_c": round(max(b["temps"]), 1),
                    "max_rain_probability": round(max(b["pops"]), 2),
                    "conditions": ", ".join(c for c in common if c),
                    "partial_day": len(b["temps"]) < 8,
                }
            )
        else:
            not_covered.append(key)
        cur += dt.timedelta(days=1)
    return {
        "ok": True,
        "location": g,
        "days": days,
        "not_covered": not_covered,
        "forecast_horizon_end": max(buckets) if buckets else None,
        "source": "OpenWeatherMap 5 day / 3 hour forecast",
        "retrieved_at": now_iso(),
    }


def _aviation_get(params: dict) -> dict:
    """AviationStack real-time /flights (included in the FREE plan).
    Free keys may not allow HTTPS - if AviationStack says so, retry once over HTTP."""
    import httpx

    key = os.getenv("AVIATIONSTACK_API_KEY", "").strip()
    if not key:
        raise ValueError("AVIATIONSTACK_API_KEY is not set")
    base = os.getenv("AVIATIONSTACK_BASE_URL", "https://api.aviationstack.com").rstrip("/")

    def get(b: str) -> dict:
        r = httpx.get(f"{b}/v1/flights", params={**params, "access_key": key}, timeout=15)
        return r.json()

    data = get(base)
    err = data.get("error") if isinstance(data, dict) else None
    if err and err.get("code") == "https_access_restricted" and base.startswith("https://"):
        data = get("http://" + base[len("https://"):])
        err = data.get("error") if isinstance(data, dict) else None
    if err:
        raise ValueError(f"AviationStack {err.get('code')}: {str(err.get('message', ''))[:150]}")
    if not isinstance(data, dict):
        raise ValueError("malformed AviationStack response")
    return data


def summarize_route_flights(rows: list) -> list[dict]:
    """One row per operating flight (codeshares dropped), sorted by departure time."""
    seen, out = set(), []
    for r in rows if isinstance(rows, list) else []:
        f = r.get("flight") or {}
        code = f.get("iata") or f.get("number")
        if f.get("codeshared") or not code or code in seen:
            continue
        seen.add(code)
        dep, arr = r.get("departure") or {}, r.get("arrival") or {}
        out.append({"flight": code, "airline": (r.get("airline") or {}).get("name") or "", "dep_time": (dep.get("scheduled") or "")[11:16],
                    "arr_time": (arr.get("scheduled") or "")[11:16], "date": r.get("flight_date"), "status": r.get("flight_status")})
    return sorted(out, key=lambda x: x["dep_time"] or "99:99")[:10]


def route_flights_today(dep_iata: str, arr_iata: str) -> dict:
    dep, arr = dep_iata.strip().upper(), arr_iata.strip().upper()
    if not (re.fullmatch(r"[A-Z]{3}", dep) and re.fullmatch(r"[A-Z]{3}", arr)):
        raise ValueError("airport codes must be 3-letter IATA codes")
    flights = summarize_route_flights(_aviation_get({"dep_iata": dep, "arr_iata": arr, "limit": 100}).get("data"))
    return {"ok": True, "route": f"{dep}->{arr}", "flights": flights, "count": len(flights),
            "source": "AviationStack real-time flights (free plan)", "retrieved_at": now_iso()}


def run_tools_server() -> None:
    """Our own MCP server (started by the app as a subprocess, stdio). Never print to stdout here."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("travel-tools")

    def safe(fn, *args) -> dict:
        try:
            return fn(*args)
        except Exception as exc:
            return {"ok": False, "error": short_error(exc)}

    @server.tool(description="Current weather for a city (geocoded with OpenWeatherMap).")
    def get_current_weather(city: str) -> dict:
        return safe(weather_current, city)

    @server.tool(
        description="Daily forecast summary (min/max temp, max rain probability, conditions) for each date "
        "between start_date and end_date (YYYY-MM-DD). Dates beyond the ~5-day forecast horizon are listed in "
        "'not_covered' - no data is invented for them."
    )
    def get_daily_forecast(city: str, start_date: str, end_date: str) -> dict:
        return safe(weather_forecast, city, start_date, end_date)

    @server.tool(
        description="Flights seen operating TODAY between two airports (IATA codes), from AviationStack's real-time "
        "endpoint (works on the free plan). A guide to which airlines fly the route and their usual times - "
        "NOT a schedule for a future date, and no fares."
    )
    def get_route_flights_today(dep_iata: str, arr_iata: str) -> dict:
        return safe(route_flights_today, dep_iata, arr_iata)

    server.run()  # stdio transport


# =============================================================================
# 7. THE THREE RETRIEVAL AGENTS' TOOLBOX (real implementation over MCP)
# =============================================================================
def source_result(source: str, status: str, reason: str = "", evidence: list | None = None, data: Any = None, cacheable: bool = True) -> dict:
    return {"source": source, "status": status, "reason": reason, "evidence": evidence or [], "data": data or {}, "cacheable": cacheable, "retrieved_at": now_iso()}


class MCPToolBox:
    def __init__(self, settings: Settings, today: dt.date | None = None):
        self.s = settings
        self.today = today or dt.date.today()

    # ---- Flight agent ------------------------------------------------------
    def flights(self, trip: TripRequest) -> dict:
        if not self.s.aviation_key:
            return source_result("flights", "unavailable", "AVIATIONSTACK_API_KEY is not set.", cacheable=False)
        calls, labels = [], []
        for fn in trip.flight_numbers[:2]:
            calls.append((["get_flight_status"], {"flight_iata": fn.replace(" ", "").upper()}))
            labels.append(("status", fn, None))
        can_schedule = (
            self.s.aviation_plan == "basic" and trip.origin_iata and trip.destination_iata and trip.start_date
            and dt.date.fromisoformat(trip.start_date) > self.today
        )
        if can_schedule:
            for dep, arr, day in ((trip.origin_iata, trip.destination_iata, trip.start_date), (trip.destination_iata, trip.origin_iata, trip.end_date)):
                if day:
                    calls.append(
                        (["future_flights_arrival_departure_schedule"],
                         {"airport_iata_code": dep, "schedule_type": "departure", "airline_iata": "", "date": day, "number_of_flights": 100})
                    )
                    labels.append(("schedule", f"{dep}->{arr}", (arr, day)))
        route_today = self.s.aviation_plan == "free" and bool(trip.origin_iata and trip.destination_iata)
        if not calls and not route_today:
            return source_result("flights", "unavailable", "Origin/destination airport codes are missing. "
                                 "Mention a flight number (e.g. 'AI 380') to check its live status. Fares are never provided by AviationStack.")
        evidence, notes, statuses = [], [], []
        if route_today:  # free-plan alternative: who flies this route today, and when
            out = call_mcp("travel-tools", [(["get_route_flights_today"], {"dep_iata": trip.origin_iata, "arr_iata": trip.destination_iata})], self.s)[0]
            try:
                payload = json.loads(out.get("text") or "{}") if out["status"] == "ok" else {"ok": False, "error": out.get("error")}
            except ValueError:
                payload = {"ok": False, "error": "malformed response"}
            route = f"{trip.origin_iata}->{trip.destination_iata}"
            if payload.get("ok") and payload.get("flights"):
                statuses.append("ok")
                notes.append(f"Free plan: these are flights seen on {route} TODAY - a guide to airlines and usual times, not a schedule for your date. No fares.")
                for f in payload["flights"][:6]:
                    evidence.append({"source_type": "flight_route_today", "provider": "AviationStack",
                                     "title": f"{f['airline']} {f['flight']} {route} dep {f['dep_time'] or '?'} arr {f['arr_time'] or '?'}",
                                     "url": None, "applies_to": f"seen {f.get('date') or 'today'} (pattern only)",
                                     "snippet": "live data for today; check the airline for your travel date"})
            elif payload.get("ok"):
                statuses.append("empty")
                notes.append(f"No flights seen on {route} today (there may be no direct flights).")
            else:
                err = str(payload.get("error") or "unknown error")
                statuses.append("unavailable" if any(k in err.lower() for k in PLAN_RESTRICTED) else "error")
                notes.append(f"route lookup: {err[:160]}")
        outcomes = call_mcp("aviationstack", calls, self.s) if calls else []
        for (kind, label, extra), out in zip(labels, outcomes):
            if out["status"] != "ok":
                statuses.append(out["status"])
                notes.append(f"{label}: {out.get('error')}")
                continue
            parsed = parse_aviation(out["text"])
            statuses.append(parsed["status"])
            if parsed["reason"]:
                notes.append(f"{label}: {parsed['reason']}")
            rows = parsed["data"]
            if kind == "schedule":
                arr, day = extra
                rows = [r for r in rows if str(r.get("arrival_airport_code", "")).upper() == arr]
                if not rows and parsed["status"] == "ok":
                    statuses[-1] = "empty"
                    notes.append(f"{label}: no direct flights found on {day}")
                for r in rows[:5]:
                    evidence.append({
                        "source_type": "flight_schedule", "provider": "AviationStack",
                        "title": f"{r.get('airline')} {r.get('flight_number')} {label} dep {r.get('departure_scheduled_time')} arr {r.get('arrival_scheduled_time')}",
                        "url": None, "applies_to": day, "snippet": f"aircraft: {r.get('aircraft') or 'n/a'}",
                    })
            else:
                for r in rows[:1]:
                    evidence.append({
                        "source_type": "flight_status", "provider": "AviationStack",
                        "title": f"{r.get('flight_number')} {r.get('departure_iata')}->{r.get('arrival_iata')}: {r.get('flight_status')}",
                        "url": None, "applies_to": r.get("flight_date") or "today",
                        "snippet": f"scheduled dep {r.get('departure_scheduled')} / arr {r.get('arrival_scheduled')} (live status, not your travel date)",
                    })
        status = "ok" if evidence else ("unavailable" if "unavailable" in statuses else "error" if "error" in statuses else "empty")
        return source_result("flights", status, "; ".join(notes), number_evidence(evidence, "F"), cacheable=status != "error")

    # ---- Weather agent -----------------------------------------------------
    def weather(self, trip: TripRequest) -> dict:
        if not self.s.owm_key:
            return source_result("weather", "unavailable", "OPENWEATHER_API_KEY is not set.", cacheable=False)
        city = trip.destination_city + (f",{trip.destination_country}" if trip.destination_country else "")
        outs = call_mcp("travel-tools", [(["get_current_weather"], {"city": city}),
                                    (["get_daily_forecast"], {"city": city, "start_date": trip.start_date, "end_date": trip.end_date})], self.s)
        payloads = []
        for o in outs:
            try:
                p = json.loads(o.get("text") or "{}")
            except ValueError:
                p = {"ok": False, "error": "malformed weather response"}
            if o["status"] != "ok":
                p = {"ok": False, "error": o.get("error") or o["status"]}
            payloads.append(p)
        cur, fc = payloads
        evidence = []
        if cur.get("ok"):
            evidence.append({"id": "W0", "source_type": "weather_current", "provider": "OpenWeatherMap", "url": None, "applies_to": "now",
                             "title": f"Now in {cur['location']['name']}: {cur['temperature_c']}°C, {cur['conditions']}", "snippet": "", "retrieved_at": cur["retrieved_at"]})
        days = fc.get("days", []) if fc.get("ok") else []
        for d in days:
            evidence.append({"id": f"W-{d['date']}", "source_type": "weather_forecast", "provider": "OpenWeatherMap", "url": None, "applies_to": d["date"],
                             "title": f"{d['date']}: {d['temp_min_c']}-{d['temp_max_c']}°C, rain chance {int(d['max_rain_probability']*100)}%, {d['conditions']}",
                             "snippet": "partial day" if d.get("partial_day") else "", "retrieved_at": fc["retrieved_at"]})
        reasons = [p["error"] for p in payloads if not p.get("ok")]
        not_covered = fc.get("not_covered", []) if fc.get("ok") else []
        if not_covered:
            reasons.append(f"{len(not_covered)} trip day(s) are beyond the ~5-day forecast horizon (no forecast invented)")
        status = "ok" if evidence else "error"
        return source_result("weather", status, "; ".join(reasons), evidence,
                             {"days": {d["date"]: d for d in days}, "not_covered": not_covered}, cacheable=status == "ok")

    # ---- Research agent (Tavily) ---------------------------------------------
    def research(self, trip: TripRequest, profile: dict) -> dict:
        if not self.s.tavily_key:
            return source_result("research", "unavailable", "TAVILY_API_KEY is not set.", cacheable=False)
        queries = research_queries(trip, profile, self.today)
        outs = call_mcp("tavily", [(["tavily_search", "tavily-search"], {"query": q, "max_results": 4, "search_depth": "basic"}) for q, _ in queries], self.s)
        evidence, notes = [], []
        for (q, label), o in zip(queries, outs):
            if o["status"] != "ok":
                notes.append(f"'{label}' search failed: {o.get('error')}")
                continue
            rows = parse_tavily(o.get("text", ""), o.get("structured"))
            if not rows and "URL:" in (o.get("text") or ""):
                notes.append(f"'{label}' search returned a malformed response")
            for r in rows:
                evidence.append({"source_type": "web_search", "provider": "Tavily", "title": r.get("title", "")[:150], "url": r["url"],
                                 "snippet": (r.get("content") or "")[:350], "applies_to": label, "published_date": r.get("published_date")})
        status = "ok" if evidence else ("error" if notes else "empty")
        return source_result("research", status, "; ".join(notes), number_evidence(evidence, "S"), cacheable=status != "error")


def number_evidence(items: list[dict], prefix: str) -> list[dict]:
    stamp = now_iso()
    for i, e in enumerate(items, 1):
        e.setdefault("id", f"{prefix}{i}")
        e.setdefault("retrieved_at", stamp)
    return items


def research_queries(trip: TripRequest, profile: dict, today: dt.date) -> list[tuple[str, str]]:
    """Up to 3 Tavily queries, shaped by the Travel DNA."""
    place = ", ".join(x for x in (trip.destination_city, trip.destination_country) if x)
    stay = profile.get("accommodation") or "well-reviewed"
    styles = " ".join(profile.get("travel_style") or [])
    extras = " ".join(x for x in (profile.get("accessibility") and "wheelchair accessible", profile.get("food") and f"{profile['food']} friendly") if x)
    likes = ", ".join((trip.interests or []) + (profile.get("favourite_activities") or [])) or "top sights"
    q = [
        (f"{stay} hotels in {place} {styles} {extras} neighbourhood guide".strip(), "stays"),
        (f"best {likes} in {place} {extras}".strip(), "things to do"),
    ]
    if trip.start_date and dt.date.fromisoformat(trip.start_date) > today + dt.timedelta(days=5):
        month = dt.date.fromisoformat(trip.start_date).strftime("%B")
        q.append((f"{place} weather in {month} what to expect rain temperature", f"typical {month} weather"))
    return q


# =============================================================================
# 8. MEMORY: Travel DNA, suggestions, conversation index (LangGraph Store)
# =============================================================================
KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def ns(user_key: str, kind: str) -> tuple[str, str, str]:
    """The ONLY way namespaces are built -> one user can never address another user's memory."""
    if not KEY_RE.match(str(user_key or "")):
        raise ValueError("Invalid traveller key")
    return ("users", user_key, kind)


def norm_pref(field: str, value: Any) -> Any:
    if PROFILE_FIELDS.get(field) == "list":
        items = value if isinstance(value, list) else re.split(r"[,;]", str(value or ""))
        return sorted({str(v).strip().lower() for v in items if str(v).strip()})
    value = str(value or "").strip()
    if field == "pace":
        v = value.lower()
        return "relaxed" if v.startswith(("relax", "slow", "easy")) else "packed" if v.startswith(("pack", "busy", "fast", "intense")) else "balanced" if v else ""
    return value


def get_profile(store, user_key: str) -> dict:
    item = store.get(ns(user_key, "profile"), "dna")
    return dict(item.value) if item else {}


def profile_values(profile: dict) -> dict:
    return {k: v["value"] for k, v in profile.items() if v.get("value") not in (None, "", [])}


def save_profile(store, user_key: str, values: dict, source: str = "form") -> dict:
    prof = get_profile(store, user_key)
    for field, value in values.items():
        if field not in PROFILE_FIELDS:
            continue
        v = norm_pref(field, value)
        if v in ("", []):
            prof.pop(field, None)
        elif prof.get(field, {}).get("value") != v:
            prof[field] = {"value": v, "source": source, "updated_at": now_iso()}
    store.put(ns(user_key, "profile"), "dna", prof)
    return prof


def delete_profile_field(store, user_key: str, field: str) -> None:
    prof = get_profile(store, user_key)
    prof.pop(field, None)
    store.put(ns(user_key, "profile"), "dna", prof)


def clear_memory(store, user_key: str) -> None:
    store.delete(ns(user_key, "profile"), "dna")
    for s in list_suggestions(store, user_key):
        store.delete(ns(user_key, "suggestions"), s["id"])


def list_suggestions(store, user_key: str) -> list[dict]:
    return [{"id": it.key, **it.value} for it in store.search(ns(user_key, "suggestions"), limit=50)]


def add_suggestion(store, user_key: str, field: str, value: Any, origin: str, quote: str = "") -> Optional[dict]:
    if field not in PROFILE_FIELDS:
        return None
    v = norm_pref(field, value)
    if v in ("", []):
        return None
    current = profile_values(get_profile(store, user_key)).get(field)
    if current == v or (isinstance(current, list) and isinstance(v, list) and set(v) <= set(current)):
        return None  # already known
    for s in list_suggestions(store, user_key):
        if s["field"] == field and s["value"] == v:
            return None  # already pending
    sug = {"field": field, "value": v, "origin": origin, "quote": quote[:200], "created_at": now_iso()}
    store.put(ns(user_key, "suggestions"), uuid.uuid4().hex[:12], sug)
    return sug


def accept_suggestion(store, user_key: str, sid: str) -> None:
    item = store.get(ns(user_key, "suggestions"), sid)
    if not item:
        return
    field, value = item.value["field"], item.value["value"]
    if PROFILE_FIELDS[field] == "list":
        value = sorted(set(profile_values(get_profile(store, user_key)).get(field, [])) | set(value))
    save_profile(store, user_key, {field: value}, source="accepted suggestion")
    store.delete(ns(user_key, "suggestions"), sid)


def dismiss_suggestion(store, user_key: str, sid: str) -> None:
    store.delete(ns(user_key, "suggestions"), sid)


def record_conversation(store, user_key: str, conversation_id: str, title: str) -> None:
    store.put(ns(user_key, "conversations"), conversation_id, {"title": title[:80], "updated_at": now_iso()})


def list_conversations(store, user_key: str) -> list[dict]:
    rows = [{"id": it.key, **it.value} for it in store.search(ns(user_key, "conversations"), limit=30)]
    return sorted(rows, key=lambda r: r.get("updated_at", ""), reverse=True)


def merge_profile(saved: dict, trip_prefs: dict) -> tuple[dict, list[dict], list[str]]:
    """Effective profile = this trip's wishes > saved Travel DNA. Returns (values, applied, conflicts)."""
    eff, applied, conflicts = {}, [], []
    trip_prefs = {k: norm_pref(k, v) for k, v in (trip_prefs or {}).items() if k in PROFILE_FIELDS}
    for field in PROFILE_FIELDS:
        tv, sv = trip_prefs.get(field), saved.get(field)
        if tv not in (None, "", []):
            eff[field] = tv
            applied.append({"field": field, "value": tv, "origin": "this trip"})
            if sv not in (None, "", []) and sv != tv:
                conflicts.append(f"Your Travel DNA says {field} = {fmt(sv)}, but for this trip you asked for {fmt(tv)} - using this trip's choice (your DNA is unchanged).")
        elif sv not in (None, "", []):
            eff[field] = sv
            applied.append({"field": field, "value": sv, "origin": "saved Travel DNA"})
    return eff, applied, conflicts


def fmt(v: Any) -> str:
    return ", ".join(v) if isinstance(v, list) else str(v)


# =============================================================================
# 9. REALITY CHECK  (deterministic validator: findings + scores + alternatives)
# =============================================================================
MEAT_WORDS = ("steak", "bbq", "barbecue", "pork", "beef", "chicken", "seafood", "fish market", "lamb", "mutton", "bacon", "prawn", "crab", "lobster", "oyster", "kebab")
MOBILITY_WORDS = ("wheelchair", "walk", "mobility", "knee", "stairs", "elderly", "senior", "stroller", "pram", "crutch", "disab", "injur", "joint", "back pain", "breath")
STRENUOUS_WORDS = ("hike", "hiking", "trek", "climb", "stairs", "steep", "kayak", "rafting", "bike tour", "cycling tour")
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def _minutes(t: str) -> Optional[int]:
    m = TIME_RE.match((t or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def trip_dates(trip: TripRequest) -> list[str]:
    s, e = dt.date.fromisoformat(trip.start_date), dt.date.fromisoformat(trip.end_date)
    return [(s + dt.timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


def finding(severity: str, code: str, message: str, suggestion: str = "") -> dict:
    return {"severity": severity, "code": code, "message": message, "suggestion": suggestion}


def reality_check(itin: Itinerary, trip: TripRequest, eff: dict, evidence: dict[str, dict], weather_days: dict, contingency: float) -> tuple[list[dict], dict]:
    F: list[dict] = []
    pace = eff.get("pace") or "balanced"
    cap = PACE_CAPS.get(pace, 4)
    dates = trip_dates(trip)
    items = [(d, it) for d in itin.days for it in d.items]

    # --- dates & coverage -----------------------------------------------------
    planned = [d.date for d in itin.days]
    if len(planned) != len(dates):
        F.append(finding("error", "day_count", f"The plan has {len(planned)} day(s) but your trip is {len(dates)} day(s).", f"Plan exactly one day per date {dates[0]} to {dates[-1]}."))
    outside = [p for p in planned if p not in dates]
    if outside or len(set(planned)) != len(planned):
        F.append(finding("error", "dates", f"Plan days do not match your dates (unexpected/duplicate: {', '.join(outside) or 'duplicates'}).", "Use each trip date once."))

    # --- pace / schedule plausibility ---------------------------------------------
    pace_pen = 0
    for i, d in enumerate(itin.days):
        acts = [it for it in d.items if it.kind in ("indoor", "outdoor")]
        limit = cap - 1 if i == 0 and len(itin.days) > 1 else cap  # arrival day is lighter
        if len(acts) > limit:
            pace_pen += 25
            F.append(finding("error", "overloaded_day", f"{d.date}: {len(acts)} activities, but a {pace} pace allows {limit}{' on arrival day' if i == 0 and len(itin.days) > 1 else ''}.",
                             f"Move or drop {len(acts) - limit} activity(ies) on {d.date}."))
        timed = sorted([(m0, m1, it.title) for it in d.items if (m0 := _minutes(it.start)) is not None and (m1 := _minutes(it.end)) is not None], key=lambda x: x[0])
        for (a0, a1, at), (b0, b1, bt) in zip(timed, timed[1:]):
            if b0 < a1:
                pace_pen += 15
                F.append(finding("error", "overlap", f"{d.date}: '{at}' overlaps with '{bt}'.", "Shift the later item or shorten the earlier one."))
        busy = sum(max(0, m1 - m0) for m0, m1, _ in timed) / 60
        if busy > 11:
            pace_pen += 10
            F.append(finding("warning", "long_day", f"{d.date}: about {busy:.0f} scheduled hours.", "Add rest time between activities."))

    # --- budget ----------------------------------------------------------------
    budget_score = None
    stay_cost = itin.stays[0].est_total_cost if itin.stays else 0.0
    total = sum(it.est_cost for _, it in items) + stay_cost
    if trip.budget_amount and itin.stays and not stay_cost:
        F.append(finding("error", "missing_stay_cost", f"No cost was estimated for the stay ('{itin.stays[0].name}'), so the budget check would be misleading.",
                         "Give the first stay an est_total_cost for the whole trip."))
    airfare = 0.0
    if trip.budget_includes_flights:
        if itin.estimated_airfare:
            airfare = itin.estimated_airfare
            F.append(finding("info", "airfare_estimate", f"Airfare ({itin.currency} {airfare:,.0f}) is an LLM estimate - AviationStack provides no fares.", "Check real fares before relying on this budget."))
        else:
            F.append(finding("warning", "airfare_unknown", "Your budget includes flights, but no fare data is available.", "The budget check below covers ground costs only."))
    if trip.budget_amount:
        cur_budget = (trip.budget_currency or "").upper()
        if cur_budget and itin.currency and cur_budget != itin.currency.upper():
            F.append(finding("warning", "currency", f"Budget is in {cur_budget} but costs are in {itin.currency}; no currency conversion is done.", "Ask for costs in your budget currency."))
        else:
            needed = (total + airfare) * (1 + contingency)
            if needed > trip.budget_amount:
                over = needed - trip.budget_amount
                budget_score = max(0, round(100 - over / trip.budget_amount * 200))
                F.append(finding("error", "over_budget",
                                 f"Estimated {itin.currency} {total + airfare:,.0f} + {int(contingency*100)}% contingency = {needed:,.0f} exceeds your budget of {trip.budget_amount:,.0f} by {over:,.0f}.",
                                 budget_alternatives(itin, trip, total, airfare, contingency, over)))
            else:
                budget_score = 100
                F.append(finding("info", "budget_ok", f"Estimated {itin.currency} {total + airfare:,.0f} (stay {stay_cost:,.0f} + activities/food {total - stay_cost:,.0f}"
                                 f"{f' + airfare {airfare:,.0f}' if airfare else ''}) + {int(contingency*100)}% contingency = {needed:,.0f}, within your {trip.budget_amount:,.0f} budget."))

    # --- weather: Rain Plan B ------------------------------------------------------
    weather_score = None
    if weather_days:
        weather_score = 100
        for d in itin.days:
            w = weather_days.get(d.date)
            if not w:
                continue
            risky = w["max_rain_probability"] >= 0.6 or w["temp_max_c"] >= 35 or w["temp_min_c"] <= 0
            if not risky:
                continue
            for it in d.items:
                if it.kind == "outdoor" and not (it.rain_alternative or "").strip():
                    weather_score -= 20
                    F.append(finding("error", "no_plan_b", f"{d.date}: '{it.title}' is outdoors but the forecast shows {int(w['max_rain_probability']*100)}% rain chance, {w['temp_min_c']}-{w['temp_max_c']}°C.",
                                     "Add an indoor Plan B (rain_alternative) or move it to a better day."))
        weather_score = max(0, weather_score)

    # --- grounding: citations & preference references -----------------------------------
    cited, bad_ids, bad_prefs = 0, set(), set()
    things = [(it.title, it.evidence_ids, it.preference_fields, it.cost_basis) for _, it in items] + [(s.name, s.evidence_ids, s.preference_fields, "estimate") for s in itin.stays]
    for title, ev_ids, prefs, basis in things:
        valid = [e for e in ev_ids if e in evidence]
        bad_ids |= set(ev_ids) - set(evidence)
        cited += bool(valid)
        bad_prefs |= {p for p in prefs if p not in eff}
        if basis == "source" and not valid:
            F.append(finding("error", "unsupported_price", f"'{title}' claims a source-backed price but cites no valid source.", "Mark the cost as an estimate or cite the source."))
    if bad_ids:
        F.append(finding("error", "bad_citation", f"Cited source id(s) that do not exist: {', '.join(sorted(bad_ids))}.", "Cite only the provided evidence ids."))
    if bad_prefs:
        F.append(finding("error", "bad_preference_ref", f"Claims to match preferences you don't have: {', '.join(sorted(bad_prefs))}.", "Only reference fields of your effective Travel DNA."))
    grounding = round(100 * cited / len(things)) if things else 0
    grounding = max(0, grounding - 25 * len(bad_ids))

    # --- preference conflicts ----------------------------------------------------
    pref_pen = 0
    food = str(eff.get("food", "")).lower()
    access = str(eff.get("accessibility", "")).lower()
    if not any(w in access for w in MOBILITY_WORDS):
        access = ""  # e.g. "car ride" is a transport wish, not a mobility need -> don't block hikes
    avoid = eff.get("avoid") or []
    for d, it in items:
        t = it.title.lower()
        if ("vegetarian" in food or "vegan" in food or "jain" in food) and any(w in t for w in MEAT_WORDS):
            pref_pen += 20
            F.append(finding("error", "diet_conflict", f"{d.date}: '{it.title}' may not suit your {food} diet.", "Swap for a vegetarian-friendly option."))
        if access and any(w in t for w in STRENUOUS_WORDS):
            pref_pen += 20
            F.append(finding("error", "access_conflict", f"{d.date}: '{it.title}' may be difficult given '{access}'.", "Choose an accessible alternative."))
        for a in avoid:
            if a and a in t:
                pref_pen += 20
                F.append(finding("error", "avoid_conflict", f"{d.date}: '{it.title}' includes '{a}', which you asked to avoid.", f"Remove anything involving '{a}'."))
    with_reason = sum(1 for _, it in items if it.why.strip() and any(p in eff for p in it.preference_fields))
    pref_fit = (round(100 * with_reason / len(items)) if items else 0) if eff else None
    if pref_fit is not None:
        pref_fit = max(0, pref_fit - pref_pen)

    scores = {"budget_fit": budget_score, "pace_fit": max(0, 100 - pace_pen), "weather_ready": weather_score, "grounding": grounding, "preference_fit": pref_fit}
    present = [v for v in scores.values() if v is not None]
    scores["overall"] = round(sum(present) / len(present)) if present else 0
    return F, scores


def budget_alternatives(itin: Itinerary, trip: TripRequest, total: float, airfare: float, contingency: float, over: float) -> str:
    """Computed (not invented) ways to get back within budget."""
    opts = []
    costly = sorted([(it.est_cost, d.date, it.title) for d in itin.days for it in d.items if it.est_cost > 0], reverse=True)
    if itin.stays and itin.stays[0].est_total_cost:
        costly.insert(0, (itin.stays[0].est_total_cost * 0.4, "whole trip", f"a cheaper stay than {itin.stays[0].name} (~40% less)"))
    saved, cut = 0.0, []
    for cost, day, title in costly:
        if saved * (1 + contingency) >= over:
            break
        saved += cost
        cut.append(f"'{title}' ({day})")
    if cut and saved * (1 + contingency) >= over:
        opts.append(f"drop or replace {', '.join(cut[:3])} (saves ~{saved:,.0f})")
    n = len(itin.days) or 1
    per_day = total / n
    if per_day > 0:
        fit_days = int((trip.budget_amount / (1 + contingency) - airfare) // per_day)
        if 0 < fit_days < n:
            opts.append(f"shorten the trip to {fit_days} day(s)")
    opts.append(f"raise the budget to ~{(total + airfare) * (1 + contingency):,.0f}")
    return "Options: " + "; or ".join(opts) + "."


# =============================================================================
# 10. THE LANGGRAPH WORKFLOW
# =============================================================================
class State(TypedDict, total=False):
    user_key: str
    conversation_id: str
    user_input: str
    feedback: Optional[dict]
    chat: Annotated[list[dict], operator.add]   # conversation log (persists via the checkpointer)
    trip: dict
    clarification: Optional[str]
    effective: dict
    applied: list
    conflicts: list
    new_suggestions: list
    flights: dict
    weather: dict
    research: dict
    itinerary: Optional[dict]
    prev_itinerary: Optional[dict]
    feedback_note: Optional[str]
    findings: list
    draft_findings: list
    scores: dict
    revision: int
    needs_revision: bool
    answer: str


@dataclass
class Deps:
    """Runtime dependencies injected per run (real or fake) - never stored in checkpoints."""
    llm: Callable[[str, str], str]
    tools: Any
    settings: Settings
    today: dt.date
    learn_from_chat: bool = True


INTAKE_SYSTEM = """You are the TRIP INTAKE agent of a personal travel planner. Today is {today}.
Turn the user's latest message into a structured trip request.
- If a PREVIOUS TRIP is given, the message usually modifies it: return the FULL updated trip and keep unchanged fields.
- Dates: YYYY-MM-DD, resolved against today. If the user is vague ("mid-December"), pick a date and add an assumption.
- destination_city must be a city; for a country/region pick its usual base city and add an assumption.
- origin_iata / destination_iata: main airport IATA codes only if you are confident, else null.
- budget_amount is a number in budget_currency (ISO code; "2 lakhs" = 200000 INR). budget_includes_flights true/false/null.
- trip_preferences: wishes for THIS trip only, keys from {fields}; pace must be relaxed|balanced|packed.
- lasting_preferences: ONLY explicit lasting statements about the user ("I'm vegetarian", "I always travel slowly",
  "I use a wheelchair"). Never guess. field must be one of {fields}. Include the user's words in quote.
- flight_numbers: only if the user wrote flight numbers (e.g. "AI 380").
- assumptions: list every guess you made.
- clarification_question: ONLY if the destination or the dates/duration are missing and cannot be taken from the previous trip."""

PLANNER_SYSTEM = """You are the ITINERARY PLANNER agent. Build a realistic day-by-day plan personalised to the traveller.
Hard rules:
1. Exactly one day per trip date given, in order. Times are 24h "HH:MM"; items in a day must not overlap.
2. Respect the activity cap per day (count only kind indoor/outdoor; meals, travel and rest are free). Arrival day: one fewer.
3. est_cost = realistic total cost for ALL travellers in the plan currency (use the budget currency): tickets, meals, local
   transport. cost_basis="estimate" unless a cited source states the price (then "source" and cite it).
   Each stay needs est_total_cost = the whole stay for all travellers (the FIRST stay is your main pick and counts in the budget).
   Never invent flight prices; estimated_airfare only if the budget includes flights, as a rough estimate.
4. Cite evidence ONLY with the ids provided (e.g. S2, W-2026-12-12, F1). Cite a source for every stay and for every activity,
   restaurant or place that appears in the evidence; leave evidence_ids empty only if nothing supports it.
   Use kind "meal" for meals and "travel" for transfers. If flight evidence (F ids) exists, plan the arrival day around those times.
5. preference_fields: list ONLY keys from the traveller's effective Travel DNA that the item really serves, and say how in "why"
   (one short sentence, e.g. "Slow morning start matches your relaxed pace").
6. On days where the forecast shows rain chance >= 60% or extreme temperatures, every outdoor item needs a concrete indoor
   rain_alternative ("Rain Plan B").
7. Respect diet, accessibility and "avoid" preferences. Stays: 1-3 options grounded in the stays evidence where possible.
8. personalization_note: 2-3 sentences explaining how the plan reflects the traveller's DNA and this trip's wishes.
9. Put every assumption you make in assumptions."""


def _trip(state: State) -> TripRequest:
    return TripRequest(**(state.get("trip") or {}))


def complete_trip(trip: TripRequest, prev: Optional[TripRequest]) -> TripRequest:
    """Fill gaps from the previous trip, derive end/duration, clean codes."""
    if prev:
        merged = prev.model_dump()
        new = {k: v for k, v in trip.model_dump().items() if v not in (None, [], {}, "")}
        # keep dates and duration consistent when only one of them changed
        changed_dates = any(k in new and new[k] != merged.get(k) for k in ("start_date", "end_date"))
        changed_duration = "duration_days" in new and new["duration_days"] != merged.get("duration_days")
        if changed_duration and "end_date" not in new:
            merged["end_date"] = None
        elif changed_dates and "duration_days" not in new:
            merged["duration_days"] = None
        merged.update(new)
        trip = TripRequest(**merged)
        trip.assumptions = list(dict.fromkeys(trip.assumptions))
    for code in ("origin_iata", "destination_iata"):
        v = (getattr(trip, code) or "").strip().upper()
        setattr(trip, code, v if re.fullmatch(r"[A-Z]{3}", v) else None)
    if trip.budget_currency:
        trip.budget_currency = trip.budget_currency.strip().upper()[:3]
    try:
        start = dt.date.fromisoformat(trip.start_date) if trip.start_date else None
        end = dt.date.fromisoformat(trip.end_date) if trip.end_date else None
    except ValueError:
        start = end = None
        trip.start_date = trip.end_date = None
    if start and not end and trip.duration_days:
        end = start + dt.timedelta(days=trip.duration_days - 1)
    if start and end:
        days = (end - start).days + 1
        if trip.duration_days and trip.duration_days != days and days > 0:
            trip.assumptions.append(f"Used your dates ({days} days) instead of the stated {trip.duration_days} days.")
        trip.duration_days = days
        trip.end_date = end.isoformat()
    if not trip.travellers:
        trip.travellers = 1
        trip.assumptions.append("Assumed 1 traveller.")
    return trip


def trip_problem(trip: TripRequest, today: dt.date) -> Optional[str]:
    if not trip.destination_city:
        return "Where would you like to go? (a city works best)"
    if not trip.start_date:
        return f"When does your trip to {trip.destination_city} start, and for how many days?"
    if not trip.end_date:
        return f"How many days will you spend in {trip.destination_city}?"
    start, end = dt.date.fromisoformat(trip.start_date), dt.date.fromisoformat(trip.end_date)
    if start < today:
        return f"The start date {trip.start_date} is in the past. Which future date should I use?"
    if end < start:
        return "The trip ends before it starts - could you check the dates?"
    if (end - start).days + 1 > MAX_TRIP_DAYS:
        return f"I can plan up to {MAX_TRIP_DAYS} days at a time. Could you shorten the trip or split it?"
    return None


FEEDBACK_LABELS = {"too_busy": "too busy", "too_relaxed": "too relaxed", "too_expensive": "too expensive"}


def apply_feedback(trip: TripRequest, fb: dict, eff_before: dict) -> tuple[TripRequest, str, list[tuple[str, Any]]]:
    """Feedback changes THIS trip immediately and proposes (never applies) a DNA change."""
    tag, day = fb.get("tag"), fb.get("day")
    where = f" on {day}" if day else ""
    prefs = dict(trip.trip_preferences)
    suggestions: list[tuple[str, Any]] = []
    pace = eff_before.get("pace") or "balanced"
    if tag in ("too_busy", "too_relaxed"):
        idx = PACE_ORDER.index(pace) + (-1 if tag == "too_busy" else 1)
        new_pace = PACE_ORDER[max(0, min(idx, 2))]
        if day is None:
            prefs["pace"] = new_pace
            suggestions.append(("pace", new_pace))
        note = f"The traveller said the plan is {FEEDBACK_LABELS[tag]}{where}. " + (
            f"Make {day} {'lighter' if tag == 'too_busy' else 'fuller'}; keep other days unless needed." if day
            else f"Use a {new_pace} pace from now on.")
    elif tag == "too_expensive":
        styles = sorted(set(prefs.get("travel_style") or eff_before.get("travel_style") or []) | {"budget-conscious"})
        prefs["travel_style"] = styles
        suggestions.append(("travel_style", ["budget-conscious"]))
        note = f"The traveller said the plan is too expensive{where}. Cut costs by at least 25% (cheaper stays, free sights, local food)."
    else:
        note = f"Traveller feedback{where}: {fb.get('text', '')}"
    trip.trip_preferences = prefs
    return trip, note, suggestions


def intake_node(state: State, runtime) -> dict:
    deps: Deps = runtime.context
    store = runtime.store
    uk = state["user_key"]
    prev = _trip(state) if state.get("trip") else None
    saved = profile_values(get_profile(store, uk))
    reset = {"itinerary": None, "findings": [], "draft_findings": [], "scores": {}, "revision": 0, "needs_revision": False,
             "clarification": None, "new_suggestions": [], "feedback_note": None, "answer": "", "prev_itinerary": state.get("itinerary")}
    fb = state.get("feedback")
    new_sugs: list[dict] = []
    if fb and prev:
        eff_before, _, _ = merge_profile(saved, prev.trip_preferences)
        trip, note, props = apply_feedback(prev, fb, eff_before)
        out = {**reset, "feedback_note": note, "chat": [{"role": "user", "content": f"Feedback: {FEEDBACK_LABELS.get(fb.get('tag'), fb.get('text', ''))}" + (f" ({fb['day']})" if fb.get("day") else "")}]}
        for f, v in props:
            new_sugs.append({"field": f, "value": v, "origin": "feedback", "quote": note[:120]})
    else:
        text = (state.get("user_input") or "").strip()
        history = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in (state.get("chat") or [])[-6:] if m["role"] == "user")
        prompt = (f"PREVIOUS TRIP (may be empty): {prev.model_dump_json() if prev else '{}'}\n"
                  f"EARLIER USER MESSAGES IN THIS CONVERSATION:\n{history or '(none)'}\n\nLATEST MESSAGE:\n{text}")
        res = ask_json(deps.llm, IntakeResult, INTAKE_SYSTEM.format(today=deps.today.isoformat(), fields=", ".join(PROFILE_FIELDS)), prompt)
        trip = complete_trip(res.trip, prev)
        out = {**reset, "chat": [{"role": "user", "content": text}]}
        if deps.learn_from_chat:
            for p in res.lasting_preferences:
                if p.field in PROFILE_FIELDS:
                    new_sugs.append({"field": p.field, "value": p.value, "origin": "said in chat", "quote": p.quote or text[:120]})
        out["clarification"] = res.clarification_question if not trip.destination_city or not trip.start_date else None
    out["clarification"] = out.get("clarification") or trip_problem(trip, deps.today)
    eff, applied, conflicts = merge_profile(saved, trip.trip_preferences)
    out.update(trip=trip.model_dump(), effective=eff, applied=applied, conflicts=conflicts, new_suggestions=new_sugs)
    return out


def route_after_intake(state: State):
    return "finalize" if state.get("clarification") else ["flights", "weather", "research"]


def _retrieve(state: State, runtime, name: str, key_parts: list, fn) -> dict:
    """Shared wrapper: reuse last turn's result if inputs unchanged; never raise."""
    key = json.dumps(key_parts, sort_keys=True, default=str)
    prev = state.get(name) or {}
    if prev.get("input_key") == key and prev.get("cacheable"):
        return {name: {**prev, "reused": True}}
    try:
        res = fn()
    except MCPUnavailable as exc:
        res = source_result(name, "unavailable", str(exc), cacheable=False)
    except Exception as exc:
        res = source_result(name, "error", short_error(exc), cacheable=False)
    res["input_key"], res["reused"] = key, False
    return {name: res}


def flights_node(state: State, runtime) -> dict:
    t, d = _trip(state), runtime.context
    return _retrieve(state, runtime, "flights", [t.origin_iata, t.destination_iata, t.start_date, t.end_date, t.flight_numbers, d.settings.aviation_plan, d.today], lambda: d.tools.flights(t))


def weather_node(state: State, runtime) -> dict:
    t, d = _trip(state), runtime.context
    return _retrieve(state, runtime, "weather", [t.destination_city, t.destination_country, t.start_date, t.end_date, d.today], lambda: d.tools.weather(t))


def research_node(state: State, runtime) -> dict:
    t, d = _trip(state), runtime.context
    eff = state.get("effective") or {}
    shaping = {k: eff.get(k) for k in ("accommodation", "travel_style", "accessibility", "food", "favourite_activities")}
    return _retrieve(state, runtime, "research", [t.destination_city, t.destination_country, t.interests, shaping, (t.start_date or "")[:7]], lambda: d.tools.research(t, eff))


def all_evidence(state: State) -> dict[str, dict]:
    ev = {}
    for src in ("flights", "weather", "research"):
        for e in (state.get(src) or {}).get("evidence", []):
            ev[e["id"]] = e
    return ev


def planner_node(state: State, runtime) -> dict:
    deps: Deps = runtime.context
    trip = _trip(state)
    eff = state.get("effective") or {}
    pace = eff.get("pace") or "balanced"
    ev = all_evidence(state)
    weather = (state.get("weather") or {}).get("data", {})
    lines = [
        f"TRIP: {trip.model_dump_json(exclude={'assumptions'})}",
        f"TRIP DATES (one plan day each): {', '.join(trip_dates(trip))}",
        f"PACE: {pace} -> max {PACE_CAPS.get(pace, 4)} activities/day",
        f"PLAN CURRENCY: {trip.budget_currency or 'INR'}",
        f"EFFECTIVE TRAVEL DNA (field: value [origin]): " + json.dumps({a['field']: f"{fmt(a['value'])} [{a['origin']}]" for a in state.get('applied') or []}),
        "SOURCE STATUS: " + "; ".join(f"{s}={(state.get(s) or {}).get('status', 'n/a')}" for s in ("flights", "weather", "research")),
        "EVIDENCE (cite by id):",
        *[f"- {e['id']} [{e['source_type']}, applies to {e.get('applies_to')}] {e.get('title', '')} :: {e.get('snippet', '')[:180]}" for e in ev.values()],
    ]
    if weather.get("not_covered"):
        lines.append(f"NO FORECAST for: {', '.join(weather['not_covered'])} (do not invent weather for these dates).")
    if state.get("feedback_note"):
        lines.append(f"FEEDBACK TO ADDRESS: {state['feedback_note']}")
        if state.get("prev_itinerary"):
            lines.append(f"PREVIOUS PLAN (revise it, keep what still works): {json.dumps(state['prev_itinerary'])[:6000]}")
    errors = [f for f in state.get("findings") or [] if f["severity"] == "error"]
    if state.get("needs_revision") and errors:
        lines.append("REALITY CHECK FAILED - fix ALL of these in the new plan:")
        lines += [f"- {f['message']} {f['suggestion']}" for f in errors]
        lines.append(f"YOUR PREVIOUS DRAFT: {json.dumps(state.get('itinerary'))[:6000]}")
    itin = ask_json(deps.llm, Itinerary, PLANNER_SYSTEM, "\n".join(lines))
    if trip.budget_currency:
        itin.currency = itin.currency or trip.budget_currency
    return {"itinerary": itin.model_dump(), "needs_revision": False}


def reality_check_node(state: State, runtime) -> dict:
    deps: Deps = runtime.context
    trip = _trip(state)
    itin = Itinerary(**state["itinerary"])
    weather_days = ((state.get("weather") or {}).get("data") or {}).get("days") or {}
    findings, scores = reality_check(itin, trip, state.get("effective") or {}, all_evidence(state), weather_days, deps.settings.contingency)
    for c in state.get("conflicts") or []:
        findings.append(finding("info", "dna_override", c))
    for src in ("flights", "weather", "research"):
        r = state.get(src) or {}
        if r.get("status") not in ("ok", None):
            findings.append(finding("info" if r["status"] in ("unavailable", "empty") else "warning", f"{src}_{r['status']}", f"{src.capitalize()}: {r['status']} - {r.get('reason') or 'no details'}"))
        elif r.get("reason"):
            findings.append(finding("info", f"{src}_note", f"{src.capitalize()}: {r['reason']}"))
    has_errors = any(f["severity"] == "error" for f in findings)
    rev = state.get("revision", 0)
    out = {"findings": findings, "scores": scores}
    if rev == 0:
        out["draft_findings"] = findings
    if has_errors and rev < MAX_REVISIONS:
        out.update(needs_revision=True, revision=rev + 1)
    else:
        out["needs_revision"] = False
    return out


def route_after_check(state: State) -> str:
    return "planner" if state.get("needs_revision") else "finalize"


def finalize_node(state: State, runtime) -> dict:
    store, uk = runtime.store, state["user_key"]
    created = []
    for s in state.get("new_suggestions") or []:
        made = add_suggestion(store, uk, s["field"], s["value"], s["origin"], s.get("quote", ""))
        if made:
            created.append(made)
    trip = _trip(state)
    title = f"{trip.destination_city or 'New trip'}" + (f" · {trip.start_date}" if trip.start_date else "")
    record_conversation(store, uk, state["conversation_id"], title)
    answer = state["clarification"] if state.get("clarification") else render_markdown(state)
    return {"answer": answer, "new_suggestions": created, "chat": [{"role": "assistant", "content": answer}]}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(State, context_schema=Deps)
    for name, fn in [("intake", intake_node), ("flights", flights_node), ("weather", weather_node), ("research", research_node),
                     ("planner", planner_node), ("reality_check", reality_check_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, ["finalize", "flights", "weather", "research"])
    for n in ("flights", "weather", "research"):
        g.add_edge(n, "planner")  # fan-in: planner waits for all three
    g.add_edge("planner", "reality_check")
    g.add_conditional_edges("reality_check", route_after_check, ["planner", "finalize"])
    g.add_edge("finalize", END)
    return g


# =============================================================================
# 11. RENDERING (markdown answer - also used for the download)
# =============================================================================
KIND_ICON = {"outdoor": "🌳", "indoor": "🏛️", "meal": "🍽️", "travel": "🚆", "rest": "☕"}
STATUS_ICON = {"ok": "✅", "empty": "◻️", "unavailable": "🚫", "error": "⚠️"}
SEV_ICON = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}


def cite(ids: list[str], ev: dict) -> str:
    parts = []
    for i in ids:
        e = ev.get(i)
        if e and e.get("url"):
            parts.append(f"[{i}]({e['url']})")
        elif e:
            parts.append(f"`{i}`")
    return " " + " ".join(parts) if parts else ""


def render_markdown(state: State) -> str:
    trip = _trip(state)
    itin = Itinerary(**state["itinerary"]) if state.get("itinerary") else None
    ev = all_evidence(state)
    sc = state.get("scores") or {}
    out = [f"## {trip.destination_city} · {trip.start_date} → {trip.end_date} · {trip.travellers} traveller(s)"]
    if itin is None:
        return "\n".join(out + ["_No itinerary was produced._"])
    out.append(f"**Reality Check score: {sc.get('overall', 0)}/100**")
    if itin.summary:
        out.append(itin.summary)
    if itin.personalization_note:
        out.append(f"🧬 **How this fits your Travel DNA:** {itin.personalization_note}")
    if itin.stays:
        out.append("### Where to stay")
        for s in itin.stays:
            cost = f"≈ {itin.currency} {s.est_total_cost:,.0f} for the whole stay (estimate)" if s.est_total_cost else (s.price_note or "price: check source")
            out.append(f"- **{s.name}** ({s.area}) - {s.why} _{cost}_{cite(s.evidence_ids, ev)}")
    for i, d in enumerate(itin.days, 1):
        out.append(f"### Day {i} · {d.date}{' - ' + d.theme if d.theme else ''}")
        for it in d.items:
            cost = f"≈ {itin.currency} {it.est_cost:,.0f} (estimate)" if it.cost_basis == "estimate" else f"{itin.currency} {it.est_cost:,.0f} (from source)"
            line = f"- {it.start}-{it.end} {KIND_ICON.get(it.kind, '')} **{it.title}** · {cost if it.est_cost else 'free / included'}{cite(it.evidence_ids, ev)}"
            if it.why:
                line += f"\n  - 💡 _{it.why}_"
            if it.rain_alternative:
                line += f"\n  - ☔ Plan B: {it.rain_alternative}"
            out.append(line)
    fl = state.get("flights") or {}
    out.append(f"### Flights\n{STATUS_ICON.get(fl.get('status'), '')} {fl.get('status', 'n/a')} - {fl.get('reason') or ''}")
    for e in fl.get("evidence", []):
        out.append(f"- `{e['id']}` {e['title']} ({e['snippet']})")
    issues = [f for f in state.get("findings") or [] if f["severity"] in ("error", "warning")]
    if issues:
        out.append("### Reality Check notes")
        out += [f"- {SEV_ICON[f['severity']]} {f['message']} {f['suggestion']}" for f in issues]
    assumptions = trip.assumptions + itin.assumptions
    if assumptions:
        out.append("### Assumptions\n" + "\n".join(f"- ⓘ {a}" for a in dict.fromkeys(assumptions)))
    web = [e for e in ev.values() if e.get("url")]
    if web:
        out.append("### Sources\n" + "\n".join(f"- [{e['id']}] [{e['title'] or e['url']}]({e['url']}) - retrieved {e['retrieved_at'][:16]}" for e in web))
    out.append("_Costs marked ≈ are AI estimates. No fares, availability or bookings are confirmed._")
    return "\n\n".join(out)


# =============================================================================
# 12. PERSISTENCE + RUNNING A TURN
# =============================================================================
def db_reachable(url: str) -> bool:
    if not url:
        return False
    try:
        import psycopg

        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception as exc:
        log.warning("PostgreSQL unreachable: %s", short_error(exc))
        return False


_MEMORY_FALLBACK: dict = {}


def memory_fallback():
    """Process-wide in-memory saver/store (used when PostgreSQL is unavailable)."""
    if not _MEMORY_FALLBACK:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        _MEMORY_FALLBACK.update(saver=InMemorySaver(), store=InMemoryStore())
    return _MEMORY_FALLBACK["saver"], _MEMORY_FALLBACK["store"]


@contextmanager
def open_backend(settings: Settings, persistent: bool, fallback=None):
    """Yields (checkpointer, store). persistent=True -> PostgreSQL (errors propagate; no silent switch)."""
    if persistent:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore

        with PostgresSaver.from_conn_string(settings.database_url) as saver, PostgresStore.from_conn_string(settings.database_url) as store:
            saver.setup()
            store.setup()
            yield saver, store
    else:
        yield fallback or memory_fallback()


def thread_config(user_key: str, conversation_id: str) -> dict:
    ns(user_key, "x")  # validates the key
    if not KEY_RE.match(conversation_id):
        raise ValueError("Invalid conversation id")
    return {"configurable": {"thread_id": f"{user_key}:{conversation_id}"}, "recursion_limit": 30}


def run_turn(saver, store, deps: Deps, user_key: str, conversation_id: str, text: str = "", feedback: dict | None = None,
             on_update: Callable[[str, dict], None] | None = None) -> dict:
    graph = build_graph().compile(checkpointer=saver, store=store)
    config = thread_config(user_key, conversation_id)
    inp = {"user_key": user_key, "conversation_id": conversation_id, "user_input": text, "feedback": feedback}
    for chunk in graph.stream(inp, config, context=deps, stream_mode="updates"):
        for node, update in chunk.items():
            if on_update and isinstance(update, dict):
                on_update(node, update)
    return graph.get_state(config).values


def load_state(saver, store, user_key: str, conversation_id: str) -> dict:
    graph = build_graph().compile(checkpointer=saver, store=store)
    return graph.get_state(thread_config(user_key, conversation_id)).values or {}


def describe_update(node: str, upd: dict) -> str:
    if node == "intake":
        if upd.get("clarification"):
            return "🧭 Intake: need one more detail"
        t = upd.get("trip") or {}
        return f"🧭 Intake: {t.get('destination_city')} {t.get('start_date')} → {t.get('end_date')} · {len(upd.get('applied') or [])} DNA preference(s) applied"
    if node in ("flights", "weather", "research"):
        r = upd.get(node) or {}
        icon = {"flights": "✈️", "weather": "🌦️", "research": "🔎"}[node]
        return f"{icon} {node.capitalize()}: {r.get('status')}{' (reused from earlier)' if r.get('reused') else ''} · {len(r.get('evidence', []))} evidence" + (f" - {r['reason'][:120]}" if r.get("reason") else "")
    if node == "planner":
        return "🗓️ Planner: draft itinerary written"
    if node == "reality_check":
        errs = sum(f["severity"] == "error" for f in upd.get("findings", []))
        return f"✅ Reality Check: score {upd.get('scores', {}).get('overall')}/100, {errs} blocking issue(s)" + (" → sending back for one revision" if upd.get("needs_revision") else "")
    if node == "finalize":
        return "📦 Done"
    return node


# =============================================================================
# 13. STREAMLIT APP
# =============================================================================
STYLE_OPTIONS = ["budget-conscious", "comfort", "adventure", "culture", "food", "nature", "nightlife", "relaxed", "shopping", "family"]
ACTIVITY_OPTIONS = ["museums", "food markets", "street food", "history", "architecture", "hiking", "beaches", "nightlife", "shopping", "art galleries", "temples", "parks", "photography", "live music"]
STAY_OPTIONS = ["", "hostel", "budget hotel", "mid-range hotel", "boutique hotel", "luxury hotel", "homestay", "apartment"]
TRANSPORT_OPTIONS = ["", "public transport", "walking", "taxis / ride-hailing", "self-drive", "mix"]


def ui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Travel DNA Planner", page_icon="🧭", layout="wide")
    s = Settings.load()
    ss = st.session_state

    @st.cache_resource
    def _fallback():
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        return InMemorySaver(), InMemoryStore()

    ss.setdefault("user_key", str(uuid.uuid4()))
    ss.setdefault("conversation_id", str(uuid.uuid4()))
    if "persistent" not in ss:
        ss.persistent = db_reachable(s.database_url)

    def backend():
        return open_backend(s, ss.persistent, _fallback())

    # ---------------- sidebar: identity, memory, Travel DNA -----------------
    with st.sidebar:
        st.markdown("### 🔑 Your traveller key")
        st.code(ss.user_key, language=None)
        st.caption("Save this key to come back to your Travel DNA and trips. Keep it private - it is not a password-protected login.")
        with st.expander("Use a saved key"):
            pasted = st.text_input("Traveller key", key="paste_key")
            if st.button("Load key") and pasted:
                if KEY_RE.match(pasted.strip()):
                    ss.user_key, ss.conversation_id = pasted.strip(), str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("That doesn't look like a traveller key.")
        if ss.persistent:
            st.success("🐘 Persistent memory: PostgreSQL")
        else:
            st.warning("⚠️ Persistent memory is OFF (PostgreSQL unavailable). Your Travel DNA and trips are temporary and will be lost when the app restarts.")
            if st.button("Retry database"):
                ss.persistent = db_reachable(s.database_url)
                st.rerun()

        try:
            with backend() as (_, store):
                prof = get_profile(store, ss.user_key)
                sugs = list_suggestions(store, ss.user_key)
                convs = list_conversations(store, ss.user_key)
        except Exception as exc:
            st.error(f"Database error: {short_error(exc)}")
            prof, sugs, convs = {}, [], []
        vals = profile_values(prof)

        st.markdown("### 🧬 Travel DNA")
        with st.expander("Edit my Travel DNA", expanded=not vals):
            st.caption("Everything here is optional. The planner uses these for EVERY trip (a trip request can override them). Hover the ? icons for help.")
            with st.form("dna"):
                styles = st.multiselect("Travel style", STYLE_OPTIONS, default=[x for x in vals.get("travel_style", []) if x in STYLE_OPTIONS],
                                        help="The kind of trips you enjoy. Shapes which activities and stays are searched for.")
                pace = st.select_slider("Pace", ["", "relaxed", "balanced", "packed"], value=vals.get("pace", ""),
                                        help="Max activities per day: relaxed = 3, balanced = 4, packed = 6 (meals and rest don't count). The Reality Check enforces it.")
                stay = st.selectbox("Accommodation", STAY_OPTIONS, index=STAY_OPTIONS.index(vals["accommodation"]) if vals.get("accommodation") in STAY_OPTIONS else 0,
                                    help="The type of place you like to stay in.")
                food = st.text_input("Food preferences", vals.get("food", ""), placeholder="e.g. vegetarian, vegan, jain, non-veg",
                                     help="If you enter vegetarian / vegan / jain, plans with meat or seafood activities get flagged.")
                acc = st.text_input("Mobility / accessibility needs (only if you have one)", vals.get("accessibility", ""),
                                    placeholder="e.g. wheelchair user, can't walk long distances - leave empty if none",
                                    help="Only for PHYSICAL needs. If filled (e.g. 'limited walking'), hikes, treks and steep climbs are flagged as unsuitable. "
                                         "Transport wishes like 'car rides' belong in 'Getting around' instead.")
                tr = st.selectbox("Getting around", TRANSPORT_OPTIONS, index=TRANSPORT_OPTIONS.index(vals["transport"]) if vals.get("transport") in TRANSPORT_OPTIONS else 0,
                                  help="How you prefer to move around at the destination.")
                acts = st.multiselect("Favourite activities", sorted(set(ACTIVITY_OPTIONS) | set(vals.get("favourite_activities", []))), default=vals.get("favourite_activities", []),
                                      help="Used in the web search for things to do, and in the 'why it fits you' explanations.")
                budget = st.text_input("Usual budget range", vals.get("budget_range", ""), placeholder="e.g. ₹8k-12k per day",
                                       help="Your typical spend. The exact budget for a trip comes from your trip request.")
                avoid = st.text_input("Things you want to avoid", ", ".join(vals.get("avoid", [])), placeholder="e.g. nightclubs, long drives", help="Separate items with commas. Plans that include them get flagged by the Reality Check.")
                if st.form_submit_button("Save Travel DNA"):
                    with backend() as (_, store):
                        save_profile(store, ss.user_key, {"travel_style": styles, "pace": pace, "accommodation": stay, "food": food, "accessibility": acc,
                                                          "transport": tr, "favourite_activities": acts, "budget_range": budget, "avoid": avoid})
                    st.toast("Travel DNA saved")
                    st.rerun()
        if prof:
            for field, meta in prof.items():
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{field.replace('_', ' ')}**: {fmt(meta['value'])}  \n<small>source: {meta['source']}</small>", unsafe_allow_html=True)
                if c2.button("✕", key=f"del_{field}", help=f"Forget {field}"):
                    with backend() as (_, store):
                        delete_profile_field(store, ss.user_key, field)
                    st.rerun()
        if sugs:
            st.markdown("#### 💭 Suggested for your DNA")
            st.caption("Learned from what you said or your feedback. Not used until you accept.")
            for sg in sugs:
                st.markdown(f"**{sg['field'].replace('_', ' ')} → {fmt(sg['value'])}**  \n<small>{sg['origin']}: “{sg['quote']}”</small>", unsafe_allow_html=True)
                a, b = st.columns(2)
                if a.button("Accept", key=f"acc_{sg['id']}"):
                    with backend() as (_, store):
                        accept_suggestion(store, ss.user_key, sg["id"])
                    st.rerun()
                if b.button("Dismiss", key=f"dis_{sg['id']}"):
                    with backend() as (_, store):
                        dismiss_suggestion(store, ss.user_key, sg["id"])
                    st.rerun()
        ss.learn = st.toggle("Suggest DNA updates from my messages", value=ss.get("learn", True))
        if prof or sugs:
            if st.button("🗑️ Clear all my saved preferences"):
                with backend() as (_, store):
                    clear_memory(store, ss.user_key)
                st.rerun()

        st.markdown("### 🗂️ Trips")
        if st.button("➕ New trip", width="stretch"):
            ss.conversation_id = str(uuid.uuid4())
            st.rerun()
        for c in convs[:10]:
            label = ("▶ " if c["id"] == ss.conversation_id else "") + c["title"]
            if st.button(label, key=f"conv_{c['id']}", width="stretch"):
                ss.conversation_id = c["id"]
                st.rerun()

        with st.expander("⚙️ Service status"):
            for name, ok in [("Groq LLM", s.groq_key), ("Tavily search", s.tavily_key), ("OpenWeatherMap", s.owm_key),
                             (f"AviationStack ({s.aviation_plan} plan)", s.aviation_key), ("PostgreSQL", ss.persistent)]:
                st.write(("✅ " if ok else "❌ ") + name)
            st.caption(f"Model: {s.groq_model}")

    # ---------------- main area -----------------
    st.title("🧭 Travel DNA Planner")
    st.caption("Memory-augmented multi-agent travel assistant · LangGraph · MCP · PostgreSQL · Groq")
    if not s.groq_key:
        st.error("GROQ_API_KEY is missing. Add it to your .env file (see README) and restart.")
        st.stop()

    try:
        with backend() as (saver, store):
            state = load_state(saver, store, ss.user_key, ss.conversation_id)
    except Exception as exc:
        st.error(f"Could not load this conversation: {short_error(exc)}")
        state = {}

    if not state.get("chat"):
        st.info("Tell me about your trip, e.g. *“5 days in Singapore from Delhi starting 12 Dec, 2 adults, ₹1.5 lakh excluding flights, we love food markets and museums.”* "
                "Your saved Travel DNA is applied automatically.")
    for m in (state.get("chat") or [])[:-1] if state.get("itinerary") else state.get("chat") or []:
        with st.chat_message(m["role"]):
            st.markdown(m["content"] if m["role"] == "user" else (m["content"] if len(m["content"]) < 400 else "_(earlier version of the plan - the latest one is shown below)_"))

    if state.get("itinerary"):
        show_result(st, state, s)

    def execute(text: str = "", feedback: dict | None = None):
        llm = GroqLLM(s)
        deps = Deps(llm=llm, tools=MCPToolBox(s), settings=s, today=dt.date.today(), learn_from_chat=ss.get("learn", True))
        with st.status("Planning your trip… (usually 20-60 seconds)", expanded=True) as box:
            llm.on_wait = box.write
            try:
                with backend() as (saver, store):
                    run_turn(saver, store, deps, ss.user_key, ss.conversation_id, text, feedback, on_update=lambda n, u: box.write(describe_update(n, u)))
                box.update(label="Done", state="complete")
            except Exception as exc:
                box.update(label="Something went wrong", state="error")
                st.error(f"{short_error(exc)}  \nNothing was invented - please try again.")
                return
        st.rerun()

    if state.get("itinerary"):
        st.markdown("#### How does this plan fit you?")
        day_opts = ["whole plan"] + [d["date"] for d in state["itinerary"]["days"]]
        day = st.selectbox("Feedback applies to", day_opts, label_visibility="collapsed")
        c = st.columns(4)
        fb_day = None if day == "whole plan" else day
        if c[0].button("👍 Fits me"):
            with backend() as (_, store):
                n = sum(bool(add_suggestion(store, ss.user_key, a["field"], a["value"], "you liked a plan using it")) for a in state.get("applied", []) if a["origin"] == "this trip")
            st.toast("Thanks!" + (f" {n} preference(s) from this trip suggested for your DNA." if n else ""))
        if c[1].button("🐢 Too busy"):
            execute(feedback={"tag": "too_busy", "day": fb_day})
        if c[2].button("⚡ Too relaxed"):
            execute(feedback={"tag": "too_relaxed", "day": fb_day})
        if c[3].button("💸 Too expensive"):
            execute(feedback={"tag": "too_expensive", "day": fb_day})

    prompt = st.chat_input("Describe a trip, or ask for a change (e.g. “make it 4 days and add Sentosa”)")
    if prompt:
        execute(text=prompt[:2000])


def show_result(st, state: dict, s: Settings) -> None:
    itin = Itinerary(**state["itinerary"])
    ev = all_evidence(state)
    sc = state.get("scores") or {}
    trip = TripRequest(**state["trip"])
    with st.chat_message("assistant"):
        st.subheader(f"{trip.destination_city} · {trip.start_date} → {trip.end_date}")
        cols = st.columns(6)
        labels = [("Overall", "overall", "Average of the other scores (n/a ones are skipped)."),
                  ("Budget", "budget_fit", "100 = estimated costs + contingency fit your budget. n/a = no budget given."),
                  ("Pace", "pace_fit", "100 = no day exceeds your pace limit and no activities overlap."),
                  ("Weather", "weather_ready", "100 = every outdoor activity on a rainy/extreme day has a Plan B. n/a = trip is beyond the ~5-day forecast."),
                  ("Sourced", "grounding", "% of stays and activities backed by a retrieved source (see Sources tab). Low = more of the plan is the AI's own suggestion."),
                  ("DNA fit", "preference_fit", "% of activities that explain which of your preferences they serve, minus conflicts (e.g. diet). n/a = no Travel DNA.")]
        for col, (lab, key, tip) in zip(cols, labels):
            col.metric(lab, "n/a" if sc.get(key) is None else f"{sc[key]}", help=tip)
        tabs = st.tabs(["🗺️ Plan", "✅ Reality Check", "🔎 Sources", "🧬 Memory & assumptions"])
        with tabs[0]:
            if itin.personalization_note:
                st.info(f"🧬 {itin.personalization_note}")
            if itin.summary:
                st.write(itin.summary)
            if itin.stays:
                st.markdown("**Where to stay**")
                for stay in itin.stays:
                    cost = f"≈ {itin.currency} {stay.est_total_cost:,.0f} for the whole stay (estimate)" if stay.est_total_cost else (stay.price_note or "price: check source")
                    st.markdown(f"- **{stay.name}** ({stay.area}) - {stay.why} _{cost}_{cite(stay.evidence_ids, ev)}")
            weather_days = ((state.get("weather") or {}).get("data") or {}).get("days") or {}
            for i, d in enumerate(itin.days, 1):
                w = weather_days.get(d.date)
                wtxt = f" · 🌡️ {w['temp_min_c']}-{w['temp_max_c']}°C, ☔ {int(w['max_rain_probability']*100)}%" if w else ""
                with st.expander(f"Day {i} · {d.date} · {d.theme}{wtxt}", expanded=i == 1):
                    for it in d.items:
                        cost = ("free / included" if not it.est_cost else f"≈ {itin.currency} {it.est_cost:,.0f} (estimate)" if it.cost_basis == "estimate" else f"{itin.currency} {it.est_cost:,.0f} (from source)")
                        st.markdown(f"**{it.start}-{it.end}** {KIND_ICON.get(it.kind, '')} **{it.title}** · {cost}{cite(it.evidence_ids, ev)}")
                        if it.why:
                            st.caption(f"💡 Why it fits you: {it.why}" + (f"  ·  DNA: {', '.join(it.preference_fields)}" if it.preference_fields else ""))
                        if it.rain_alternative:
                            st.caption(f"☔ Rain Plan B: {it.rain_alternative}")
            st.download_button("⬇️ Download plan (Markdown)", render_markdown(state), file_name=f"trip_{trip.destination_city}_{trip.start_date}.md")
        with tabs[1]:
            draft_errs = sum(f["severity"] == "error" for f in state.get("draft_findings") or [])
            errs = sum(f["severity"] == "error" for f in state.get("findings") or [])
            if state.get("revision"):
                st.write(f"The first draft had **{draft_errs}** blocking issue(s); one automatic revision left **{errs}**.")
            for f in sorted(state.get("findings") or [], key=lambda f: ["error", "warning", "info"].index(f["severity"])):
                st.markdown(f"{SEV_ICON[f['severity']]} {f['message']}" + (f"  \n→ _{f['suggestion']}_" if f["suggestion"] else ""))
        with tabs[2]:
            for src in ("flights", "weather", "research"):
                r = state.get(src) or {}
                st.markdown(f"**{src.capitalize()}**: {STATUS_ICON.get(r.get('status'), '')} {r.get('status', 'n/a')}{' (reused)' if r.get('reused') else ''} - {r.get('reason') or ''}")
            rows = [{"id": e["id"], "type": e["source_type"], "provider": e["provider"], "what": e.get("title", ""), "link": e.get("url") or "",
                     "applies to": e.get("applies_to"), "retrieved (UTC)": (e.get("retrieved_at") or "")[:16]} for e in ev.values()]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True, column_config={"link": st.column_config.LinkColumn("link")})
            st.caption("✔ retrieved = has a source above · ≈ estimate = AI estimate · ⓘ assumption · 🚫 unavailable. No fares, availability or bookings are confirmed.")
        with tabs[3]:
            st.markdown("**Preferences used for this plan**")
            for a in state.get("applied") or []:
                st.markdown(f"- {a['field'].replace('_', ' ')}: **{fmt(a['value'])}** - _{a['origin']}_")
            if not state.get("applied"):
                st.caption("No preferences yet - fill in your Travel DNA in the sidebar.")
            for c in state.get("conflicts") or []:
                st.warning(c)
            assumptions = list(dict.fromkeys(trip.assumptions + itin.assumptions))
            if assumptions:
                st.markdown("**Assumptions**")
                for a in assumptions:
                    st.markdown(f"- ⓘ {a}")
            if state.get("new_suggestions"):
                st.markdown("**New suggestions (see sidebar to accept/dismiss)**")
                for sg in state["new_suggestions"]:
                    st.markdown(f"- {sg['field']} → {fmt(sg['value'])}")


# =============================================================================
# 14. COMMAND LINE: check / chat / selftest / tools-server
# =============================================================================
def cmd_check() -> int:
    s = Settings.load()
    print("Configuration:")
    for p in s.problems() or ["all keys present"]:
        print("  -", p)
    print(f"  model: {s.groq_model}   AviationStack plan: {s.aviation_plan}")
    print("PostgreSQL:", "reachable" if db_reachable(s.database_url) else "NOT reachable (app will use temporary memory)")
    for name, needed in (("travel-tools", s.owm_key or s.aviation_key), ("aviationstack", s.aviation_key), ("tavily", s.tavily_key)):
        if not needed:
            print(f"MCP {name}: skipped (key missing)")
            continue
        try:
            print(f"MCP {name}: tools = {asyncio.run(_list_tools(name, s))}")
        except Exception as exc:
            print(f"MCP {name}: FAILED - {short_error(exc)}")
    if s.groq_key:
        try:
            print("Groq:", "OK" if "ok" in extract_json(GroqLLM(s)("Reply with JSON.", 'Return {"status":"ok"}')) else "unexpected reply")
        except Exception as exc:
            print("Groq: FAILED -", short_error(exc))
    return 0


def cmd_chat() -> int:
    s = Settings.load()
    if not s.groq_key:
        print("GROQ_API_KEY is missing.")
        return 1
    persistent = db_reachable(s.database_url)
    print("Memory:", "PostgreSQL" if persistent else "TEMPORARY (PostgreSQL unavailable)")
    key = input("Traveller key (Enter for a new one): ").strip() or str(uuid.uuid4())
    print("Your traveller key:", key)
    cid = str(uuid.uuid4())
    while True:
        text = input("\nYou (blank to quit): ").strip()
        if not text:
            return 0
        deps = Deps(GroqLLM(s), MCPToolBox(s), s, dt.date.today())
        with open_backend(s, persistent) as (saver, store):
            st_ = run_turn(saver, store, deps, key, cid, text, on_update=lambda n, u: print("  ·", describe_update(n, u)))
        print("\n" + st_.get("answer", ""))


# ---------------------------------------------------------------------------
# Tests: `python travel_assistant.py selftest`  or  `pytest travel_assistant.py`
# Everything below runs offline with a fake LLM and fake tools.
# ---------------------------------------------------------------------------
TEST_TODAY = dt.date(2026, 11, 23)


class FakeLLM:
    """Returns canned JSON. Records every prompt so tests can check what the agents were told."""

    def __init__(self, trip: dict, plans: list[dict], lasting: list | None = None, clarification: str | None = None):
        self.trip, self.plans, self.lasting, self.clarification = trip, list(plans), lasting or [], clarification
        self.prompts: list[tuple[str, str]] = []
        self.planner_calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if "TRIP INTAKE" in system:
            return json.dumps({"trip": self.trip, "lasting_preferences": self.lasting, "clarification_question": self.clarification})
        self.planner_calls += 1
        return json.dumps(self.plans[min(self.planner_calls - 1, len(self.plans) - 1)])

    def planner_prompts(self) -> list[str]:
        return [u for s, u in self.prompts if "ITINERARY PLANNER" in s]


class FakeTools:
    def __init__(self, rain_day: str | None = None, fail: set | None = None):
        self.calls = {"flights": 0, "weather": 0, "research": 0}
        self.rain_day, self.fail = rain_day, fail or set()

    def _maybe_fail(self, name):
        self.calls[name] += 1
        if name in self.fail:
            raise MCPUnavailable(f"{name} MCP server unavailable: simulated outage")

    def flights(self, trip):
        self._maybe_fail("flights")
        return source_result("flights", "unavailable", "Your AviationStack plan is 'free'.")

    def weather(self, trip):
        self._maybe_fail("weather")
        days = {d: {"date": d, "temp_min_c": 24, "temp_max_c": 31, "max_rain_probability": 0.8 if d == self.rain_day else 0.1, "conditions": "rain" if d == self.rain_day else "clear"} for d in trip_dates(trip)}
        ev = [{"id": f"W-{d}", "source_type": "weather_forecast", "provider": "OpenWeatherMap", "url": None, "applies_to": d, "title": d, "snippet": "", "retrieved_at": now_iso()} for d in days]
        return source_result("weather", "ok", "", ev, {"days": days, "not_covered": []})

    def research(self, trip, profile):
        self._maybe_fail("research")
        ev = [{"source_type": "web_search", "provider": "Tavily", "title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": "…", "applies_to": "things to do"} for i in (1, 2)]
        return source_result("research", "ok", "", number_evidence(ev, "S"))


def _trip_json(**over) -> dict:
    base = {"destination_city": "Singapore", "destination_country": "Singapore", "origin_city": "Delhi", "origin_iata": "DEL", "destination_iata": "SIN",
            "start_date": "2026-12-12", "end_date": "2026-12-14", "travellers": 2, "budget_amount": 60000, "budget_currency": "INR",
            "budget_includes_flights": False, "interests": ["museums"], "trip_preferences": {}, "assumptions": []}
    base.update(over)
    return base


def _plan(dates=("2026-12-12", "2026-12-13", "2026-12-14"), cost=3000, acts=2, outdoor_planb=True, cite="S1", prefs=(), title="Museum visit") -> dict:
    days = []
    for d in dates:
        items = [{"start": f"{10 + 2*i:02d}:00", "end": f"{11 + 2*i:02d}:00", "title": f"{title} {i}", "kind": "outdoor" if i == 0 else "indoor", "est_cost": cost,
                  "why": "Fits your relaxed pace", "preference_fields": list(prefs), "evidence_ids": [cite] if cite else [],
                  "rain_alternative": "Indoor museum" if outdoor_planb else None} for i in range(acts)]
        days.append({"date": d, "theme": "Explore", "items": items})
    return {"summary": "A plan", "personalization_note": "Relaxed mornings.", "currency": "INR", "stays": [], "days": days, "assumptions": []}


def _env(llm, tools=None):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    s = Settings.load()
    s.contingency = 0.10
    saver, store = InMemorySaver(), InMemoryStore()
    deps = Deps(llm=llm, tools=tools or FakeTools(), settings=s, today=TEST_TODAY)
    return saver, store, deps


def _keys():
    return str(uuid.uuid4()), str(uuid.uuid4())


def test_tavily_parser_handles_text_json_and_garbage():
    text = "Detailed Results:\n\nTitle: Gardens\nURL: https://a.com/x\nContent: Big park\nmore text\nFavicon: https://a.com/f.ico\n\nTitle: Museum\nURL: https://b.com\nContent: Art"
    rows = parse_tavily(text)
    assert [r["url"] for r in rows] == ["https://a.com/x", "https://b.com"] and rows[0]["content"] == "Big park more text"
    assert parse_tavily(json.dumps({"results": [{"title": "T", "url": "https://c.com", "content": "C"}]}))[0]["url"] == "https://c.com"
    assert parse_tavily("total garbage") == [] and parse_tavily("") == []


def test_aviation_envelope_classification():
    assert parse_aviation('{"ok": true, "count": 1, "data": [{"x": 1}]}')["status"] == "ok"
    assert parse_aviation('{"ok": true, "count": 0, "data": []}')["status"] == "empty"
    assert parse_aviation('{"ok": false, "error": "function_access_restricted (105): Access Restricted"}')["status"] == "unavailable"
    assert parse_aviation('{"ok": false, "error": "invalid_access_key"}')["status"] == "error"
    assert parse_aviation("<html>oops")["status"] == "error"


def test_reality_check_catches_planted_problems():
    trip = complete_trip(TripRequest(**_trip_json()), None)
    eff = {"pace": "relaxed", "food": "vegetarian", "avoid": ["casino"]}
    bad = _plan(dates=("2026-12-12", "2026-12-13"), cost=20000, acts=4, outdoor_planb=False, cite="S9", prefs=("pace", "nightlife"))
    bad["days"][1]["items"][1]["title"] = "BBQ chicken night"
    bad["days"][1]["items"][2]["title"] = "Casino tour"
    bad["days"][1]["items"][3]["start"] = "14:30"  # item 2 runs 14:00-15:00 -> overlap
    weather = {"2026-12-12": {"temp_min_c": 25, "temp_max_c": 30, "max_rain_probability": 0.9}}
    F, sc = reality_check(Itinerary(**bad), trip, eff, {"S1": {}}, weather, 0.10)
    codes = {f["code"] for f in F}
    for c in ("day_count", "overloaded_day", "overlap", "over_budget", "no_plan_b", "bad_citation", "bad_preference_ref", "diet_conflict", "avoid_conflict"):
        assert c in codes, f"missing {c}: {codes}"
    good, sc2 = reality_check(Itinerary(**_plan(prefs=("pace",))), trip, {"pace": "relaxed"}, {"S1": {}}, weather, 0.10)
    assert not [f for f in good if f["severity"] == "error"], good
    assert sc2["overall"] > sc["overall"]


def test_budget_alternatives_are_computed():
    trip = complete_trip(TripRequest(**_trip_json(budget_amount=20000)), None)
    F, _ = reality_check(Itinerary(**_plan(cost=4000)), trip, {"pace": "relaxed"}, {"S1": {}}, {}, 0.10)
    over = next(f for f in F if f["code"] == "over_budget")
    assert "26,400" in over["message"] and "shorten the trip to 2 day(s)" in over["suggestion"] and "raise the budget" in over["suggestion"]


def test_travel_dna_is_used_in_a_new_conversation_without_repeating_it():
    llm = FakeLLM(_trip_json(), [_plan()])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    save_profile(store, uk, {"pace": "relaxed", "food": "vegetarian"})
    st_ = run_turn(saver, store, deps, uk, c1, "3 days in Singapore from 12 Dec")
    prompt = llm.planner_prompts()[0]
    assert "relaxed [saved Travel DNA]" in prompt and "vegetarian [saved Travel DNA]" in prompt
    assert {a["origin"] for a in st_["applied"]} == {"saved Travel DNA"} and st_["itinerary"]


def test_trip_wish_overrides_dna_and_is_reported():
    llm = FakeLLM(_trip_json(trip_preferences={"pace": "packed"}), [_plan()])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    save_profile(store, uk, {"pace": "relaxed"})
    st_ = run_turn(saver, store, deps, uk, c1, "packed trip please")
    assert st_["effective"]["pace"] == "packed" and st_["conflicts"] and "relaxed" in st_["conflicts"][0]
    assert profile_values(get_profile(store, uk))["pace"] == "relaxed"  # DNA unchanged


def test_statements_become_pending_suggestions_until_accepted():
    llm = FakeLLM(_trip_json(), [_plan()], lasting=[{"field": "food", "value": "vegetarian", "quote": "I'm vegetarian"}])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    run_turn(saver, store, deps, uk, c1, "I'm vegetarian, plan Singapore")
    assert get_profile(store, uk) == {}  # NOT a fact yet
    sugs = list_suggestions(store, uk)
    assert len(sugs) == 1 and sugs[0]["value"] == "vegetarian"
    accept_suggestion(store, uk, sugs[0]["id"])
    assert get_profile(store, uk)["food"]["source"] == "accepted suggestion" and list_suggestions(store, uk) == []
    deps.learn_from_chat = False
    llm.lasting = [{"field": "pace", "value": "relaxed", "quote": "I like slow trips"}]
    run_turn(saver, store, deps, uk, str(uuid.uuid4()), "again")
    assert list_suggestions(store, uk) == []  # learning switched off -> nothing suggested
    clear_memory(store, uk)
    assert get_profile(store, uk) == {}


def test_users_are_isolated():
    llm = FakeLLM(_trip_json(), [_plan()])
    saver, store, deps = _env(llm)
    a, b = _keys()
    save_profile(store, a, {"food": "vegan", "accessibility": "wheelchair user"})
    run_turn(saver, store, deps, b, str(uuid.uuid4()), "Singapore trip")
    assert "vegan" not in llm.planner_prompts()[-1] and "wheelchair" not in llm.planner_prompts()[-1]
    assert get_profile(store, b) == {} and list_conversations(store, a) == []
    for bad_key in ("../users/x", "", "abc"):
        try:
            ns(bad_key, "profile")
            raise AssertionError("invalid key accepted")
        except ValueError:
            pass


def test_clarification_spends_no_api_calls():
    tools = FakeTools()
    llm = FakeLLM(_trip_json(destination_city=None, start_date=None, end_date=None), [_plan()], clarification="Where to?")
    saver, store, deps = _env(llm, tools)
    uk, c1 = _keys()
    st_ = run_turn(saver, store, deps, uk, c1, "plan a trip")
    assert st_["answer"] and st_.get("itinerary") is None and sum(tools.calls.values()) == 0 and llm.planner_calls == 0


def test_past_dates_are_rejected():
    llm = FakeLLM(_trip_json(start_date="2026-01-01", end_date="2026-01-03"), [_plan()])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    assert "past" in run_turn(saver, store, deps, uk, c1, "Jan trip")["answer"]


def test_revision_fixes_a_bad_draft():
    llm = FakeLLM(_trip_json(), [_plan(cite="S99"), _plan()])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    st_ = run_turn(saver, store, deps, uk, c1, "Singapore")
    assert llm.planner_calls == 2 and "REALITY CHECK FAILED" in llm.planner_prompts()[1]
    assert any(f["code"] == "bad_citation" for f in st_["draft_findings"]) and not [f for f in st_["findings"] if f["severity"] == "error"]


def test_revision_is_bounded_and_remaining_issues_are_shown():
    llm = FakeLLM(_trip_json(), [_plan(cost=50000)])
    saver, store, deps = _env(llm)
    uk, c1 = _keys()
    st_ = run_turn(saver, store, deps, uk, c1, "Singapore")
    assert llm.planner_calls == 1 + MAX_REVISIONS and any(f["code"] == "over_budget" for f in st_["findings"])
    assert "exceeds your budget" in st_["answer"]


def test_rain_plan_b_is_enforced():
    llm = FakeLLM(_trip_json(), [_plan(outdoor_planb=False), _plan(outdoor_planb=True)])
    saver, store, deps = _env(llm, FakeTools(rain_day="2026-12-13"))
    uk, c1 = _keys()
    st_ = run_turn(saver, store, deps, uk, c1, "Singapore")
    assert any(f["code"] == "no_plan_b" for f in st_["draft_findings"]) and "Plan B" in st_["answer"]


def test_one_failed_source_still_gives_a_plan():
    tools = FakeTools(fail={"weather"})
    llm = FakeLLM(_trip_json(), [_plan()])
    saver, store, deps = _env(llm, tools)
    uk, c1 = _keys()
    st_ = run_turn(saver, store, deps, uk, c1, "Singapore")
    assert st_["weather"]["status"] == "unavailable" and st_["itinerary"] and st_["research"]["status"] == "ok"
    assert "weather=unavailable" in llm.planner_prompts()[0]


def test_followup_keeps_context_and_feedback_reuses_data():
    tools = FakeTools()
    llm = FakeLLM(_trip_json(), [_plan()])
    saver, store, deps = _env(llm, tools)
    uk, c1 = _keys()
    run_turn(saver, store, deps, uk, c1, "3 days Singapore")
    # follow-up: the LLM only returns the changed field; the rest comes from the checkpoint
    llm.trip = {"end_date": "2026-12-13", "trip_preferences": {}}
    llm.plans = [_plan(dates=("2026-12-12", "2026-12-13"))]
    llm.planner_calls = 0
    st2 = run_turn(saver, store, deps, uk, c1, "make it 2 days")
    assert st2["trip"]["destination_city"] == "Singapore" and st2["trip"]["duration_days"] == 2
    assert "PREVIOUS TRIP" in llm.prompts[-2][1] and '"Singapore"' in llm.prompts[-2][1]
    calls_before = dict(tools.calls)
    llm.planner_calls = 0
    st3 = run_turn(saver, store, deps, uk, c1, feedback={"tag": "too_busy", "day": None})
    assert tools.calls == calls_before, "feedback must reuse retrieved data"
    assert st3["effective"]["pace"] == "relaxed" and "FEEDBACK TO ADDRESS" in llm.planner_prompts()[-1]
    assert any(s["field"] == "pace" and s["value"] == "relaxed" for s in list_suggestions(store, uk))
    assert profile_values(get_profile(store, uk)) == {}  # feedback never silently edits the DNA
    assert len(st3["chat"]) == 6


def test_travel_tools_mcp_server_over_real_stdio():
    """Starts our real MCP server subprocess; OpenWeatherMap and AviationStack are replaced by a local stub HTTP server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    base_ts = int(dt.datetime(2026, 12, 12, tzinfo=dt.timezone.utc).timestamp())

    class Stub(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/v1/flights"):
                body = {"data": [{"flight": {"iata": "6E5321"}, "airline": {"name": "IndiGo"}, "flight_date": "2026-11-23",
                                  "departure": {"scheduled": "2026-11-23T09:10:00+00:00"}, "arrival": {"scheduled": "2026-11-23T10:20:00+00:00"}}]}
            elif self.path.startswith("/geo/1.0/direct"):
                body = [{"name": "Singapore", "country": "SG", "lat": 1.29, "lon": 103.85}]
            elif self.path.startswith("/data/2.5/weather"):
                body = {"main": {"temp": 29, "feels_like": 33, "humidity": 80}, "weather": [{"description": "light rain"}], "wind": {"speed": 3}}
            else:
                body = {"city": {"timezone": 28800}, "list": [{"dt": base_ts + h * 3600, "main": {"temp": 26 + h % 5}, "pop": 0.7 if h < 24 else 0.1, "weather": [{"description": "rain" if h < 24 else "clouds"}]} for h in range(0, 48, 3)]}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    keys = ("OPENWEATHER_BASE_URL", "OPENWEATHER_API_KEY", "AVIATIONSTACK_BASE_URL", "AVIATIONSTACK_API_KEY", "AVIATIONSTACK_PLAN")
    old = {k: os.environ.get(k) for k in keys}
    stub = f"http://127.0.0.1:{httpd.server_port}"
    os.environ.update(OPENWEATHER_BASE_URL=stub, OPENWEATHER_API_KEY="test-key", AVIATIONSTACK_BASE_URL=stub, AVIATIONSTACK_API_KEY="test-key", AVIATIONSTACK_PLAN="free")
    try:
        s = Settings.load()
        assert asyncio.run(_list_tools("travel-tools", s)) == ["get_current_weather", "get_daily_forecast", "get_route_flights_today"]
        trip = complete_trip(TripRequest(**_trip_json(start_date="2026-12-12", end_date="2026-12-20")), None)
        res = MCPToolBox(s, today=TEST_TODAY).weather(trip)
        assert res["status"] == "ok", res
        assert res["data"]["days"]["2026-12-12"]["max_rain_probability"] == 0.7
        assert "2026-12-20" in res["data"]["not_covered"] and "beyond" in res["reason"]
        assert res["evidence"][0]["id"] == "W0"
        fl = MCPToolBox(s, today=TEST_TODAY).flights(trip)  # free-plan route alternative, through the real MCP server
        assert fl["status"] == "ok" and "IndiGo 6E5321 DEL->SIN dep 09:10" in fl["evidence"][0]["title"] and "TODAY" in fl["reason"]
    finally:
        httpd.shutdown()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_missing_keys_are_reported_not_crashed():
    s = Settings.load()
    s.tavily_key = s.owm_key = s.aviation_key = ""
    tb = MCPToolBox(s, today=TEST_TODAY)
    trip = complete_trip(TripRequest(**_trip_json()), None)
    assert tb.research(trip, {})["status"] == tb.weather(trip)["status"] == tb.flights(trip)["status"] == "unavailable"
    s.aviation_key, s.aviation_plan = "x", "free"
    trip.origin_iata = None  # free plan without airport codes -> explained, no wasted API call
    res = tb.flights(trip)
    assert res["status"] == "unavailable" and "airport codes" in res["reason"]


def test_postgres_persistence_and_isolation():
    s = Settings.load()
    if not db_reachable(s.database_url):
        print("    (skipped: DATABASE_URL not reachable)")
        return
    llm = FakeLLM(_trip_json(), [_plan()])
    deps = Deps(llm=llm, tools=FakeTools(), settings=s, today=TEST_TODAY)
    a, b = _keys()
    conv = str(uuid.uuid4())
    with open_backend(s, True) as (saver, store):
        save_profile(store, a, {"pace": "relaxed"})
        run_turn(saver, store, deps, a, conv, "Singapore")
    with open_backend(s, True) as (saver, store):  # fresh connections = "app restarted"
        st_ = load_state(saver, store, a, conv)
        assert st_["trip"]["destination_city"] == "Singapore" and st_["itinerary"]
        assert profile_values(get_profile(store, a)) == {"pace": "relaxed"} and get_profile(store, b) == {}
        assert load_state(saver, store, b, conv) == {}
        clear_memory(store, a)


def test_stay_cost_counts_in_budget_and_missing_cost_is_flagged():
    trip = complete_trip(TripRequest(**_trip_json(budget_amount=25000)), None)
    plan = _plan(cost=1000)
    plan["stays"] = [{"name": "Hotel A", "est_total_cost": 20000, "evidence_ids": ["S1"]}]
    F, _ = reality_check(Itinerary(**plan), trip, {}, {"S1": {}}, {}, 0.10)
    assert any(f["code"] == "over_budget" for f in F)  # (6,000 + 20,000 stay) + 10% > 25,000
    plan["stays"][0]["est_total_cost"] = 0
    F, _ = reality_check(Itinerary(**plan), trip, {}, {"S1": {}}, {}, 0.10)
    assert any(f["code"] == "missing_stay_cost" for f in F)


def test_accessibility_only_blocks_hikes_for_real_mobility_needs():
    trip = complete_trip(TripRequest(**_trip_json()), None)
    plan = _plan(title="Hiking trail")
    F1, _ = reality_check(Itinerary(**plan), trip, {"accessibility": "car ride"}, {"S1": {}}, {}, 0.10)
    F2, _ = reality_check(Itinerary(**plan), trip, {"accessibility": "limited walking"}, {"S1": {}}, {}, 0.10)
    assert not any(f["code"] == "access_conflict" for f in F1) and any(f["code"] == "access_conflict" for f in F2)


def test_route_flights_summary_drops_codeshares_and_sorts():
    rows = [
        {"flight": {"iata": "6E5321"}, "airline": {"name": "IndiGo"}, "departure": {"scheduled": "2026-11-25T09:10:00+00:00"}, "arrival": {"scheduled": "2026-11-25T10:20:00+00:00"}, "flight_date": "2026-11-25"},
        {"flight": {"iata": "AI657"}, "airline": {"name": "Air India"}, "departure": {"scheduled": "2026-11-25T06:05:00+00:00"}, "arrival": {"scheduled": "2026-11-25T07:15:00+00:00"}, "flight_date": "2026-11-25"},
        {"flight": {"iata": "UA8000", "codeshared": {"flight_iata": "ai657"}}, "airline": {"name": "United"}, "departure": {}, "arrival": {}},
        {"flight": {"iata": "AI657"}, "airline": {"name": "Air India"}, "departure": {}, "arrival": {}},
    ]
    out = summarize_route_flights(rows)
    assert [f["flight"] for f in out] == ["AI657", "6E5321"] and out[0]["dep_time"] == "06:05"
    assert summarize_route_flights("garbage") == []


def test_groq_rate_limit_waits_then_retries():
    class RL(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("Rate limit reached. Please try again in 2.5s.")
            self.response = type("R", (), {"headers": {}})()

    class Client:
        calls = 0

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    Client.calls += 1
                    if Client.calls == 1:
                        raise RL()
                    return type("Resp", (), {"choices": [type("C", (), {"message": type("M", (), {"content": '{"ok": 1}'})()})()]})()

    waits, notes = [], []
    llm = GroqLLM(Settings.load(), client=Client, sleep=waits.append)
    llm.on_wait = notes.append
    assert llm("s", "u") == '{"ok": 1}' and waits == [3.5] and notes and Client.calls == 2


def cmd_selftest() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {short_error(exc)}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("tools-server", "weather-server"):
        run_tools_server()
        return 0
    if cmd in ("check", "chat", "selftest"):
        return {"check": cmd_check, "chat": cmd_chat, "selftest": cmd_selftest}[cmd]()
    print(__doc__)
    return 0


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("tools-server", "weather-server", "check", "chat", "selftest"):
        sys.exit(main())
    elif _running_in_streamlit():
        ui()
    else:
        sys.exit(main())