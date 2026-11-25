# travel-dna-planner

A travel planning assistant built with LangGraph, where several agents work together to put a trip plan together. It pulls live data through MCP servers, stores each user's travel preferences in PostgreSQL, and checks every itinerary against a set of rules before it is shown.

A request such as "4 days in Goa from Mumbai starting 20 Dec, 2 adults, 50k budget" returns a day-by-day plan that takes the saved preferences into account, shows where each piece of information came from, and flags anything that doesn't add up (budget, pace, weather, and so on).

Stack: Python, LangGraph, MCP, PostgreSQL, Groq, Tavily, AviationStack, OpenWeatherMap, Streamlit.

## Why I built this

Most LLM travel planners I tried had the same three problems:

- They make up details like prices, flight times and weather.
- They don't remember anything about the user, so things like "I'm vegetarian, I prefer a slow pace" have to be repeated every time.
- They never check their own output, which leads to six activities on a "relaxed" day, a beach walk during a storm, or a plan that is way over budget.

This project tries to fix all three. Live data comes from real APIs, preferences are stored in a database, and a rule-based checker reviews the plan before it reaches the user.

## How it works

The workflow is a LangGraph state graph with five stages:

```
User request (Streamlit)
        |
        v
1. INTAKE agent (LLM)
     - turns the text into structured trip details (city, dates, budget, travellers)
     - loads the saved preferences and this conversation's earlier turns (PostgreSQL)
     - missing info or past dates -> asks one question and stops (no API calls spent)
        |
        v
2. THREE RETRIEVAL AGENTS, run in parallel
     Flights   --MCP-->  AviationStack
     Weather   --MCP-->  travel-tools server (this repo)  -->  OpenWeatherMap
     Research  --MCP-->  Tavily web search (stays, things to do, typical weather)
     each reports ok / empty / unavailable / error and never crashes the run
        |
        v
3. PLANNER agent (LLM)
     itinerary as structured data: times, costs, reasons, sources, rain alternatives
        |
        v
4. REALITY CHECK (rule-based, no LLM)
     dates, pace, budget, weather, sources, preferences
     blocking issue -> back to the planner once with the list of problems
        |
        v
5. FINALIZE
     renders the plan, saves the conversation, stores preference suggestions
```

A bit more on each piece:

- **LangGraph** holds the shared state (trip details, retrieved data, the plan, the check results) and controls the flow between agents. That includes the parallel step and the single loop back from the checker to the planner.
- **MCP** is how the agents reach external data. Each source runs as its own MCP server, and the app acts as the client that discovers and calls their tools. There are three servers. The AviationStack one is an existing open-source server launched with `uvx`. The second is a small server I wrote (inside `travel_assistant.py`) for weather and route lookups. The third is Tavily's hosted server.
- **PostgreSQL** stores two things: every conversation (via LangGraph's checkpointer) and each user's saved preferences (via LangGraph's store).
- **Groq** runs the LLM. The default is `openai/gpt-oss-120b`, called twice per plan, or three times if the checker asks for a revision.
- **Streamlit** is the frontend.

## Features

### Travel DNA

This is a profile filled in once from the sidebar: travel style, pace, accommodation type, food preferences, mobility needs, preferred way of getting around, favourite activities, usual budget, and things to avoid. Pace sets a daily limit: 3 activities for relaxed, 4 for balanced, 6 for packed.

It's saved in PostgreSQL and loaded automatically for every new trip, so nothing needs to be repeated. It affects what the research agent searches for, what the planner picks, and what the checker looks for. If a specific trip asks for something different (for example, a packed schedule when the profile says relaxed), the trip request wins for that trip only. The app points out the override and leaves the profile unchanged.

### Preferences are suggested, not silently saved

If something like "I'm vegetarian" or "I always prefer homestays" comes up in the chat, it doesn't go straight into the profile. It shows up as a suggestion in the sidebar, which the user can accept or dismiss. Saved fields can be deleted one by one, the whole profile can be cleared, and suggestions from chat can be switched off altogether. I didn't want the model turning a guess into a permanent fact about someone.

### Conversation memory

Each trip is its own conversation, stored in PostgreSQL, so follow-ups like "make it 3 days" or "add a beach day" keep everything else (city, budget, number of people). Past trips are listed in the sidebar and are still there after restarting the app.

There are two separate kinds of memory here. The checkpoint stores one trip's conversation. The profile stores the user, across all trips.

### Explanations for each recommendation

Every activity and stay comes with a short reason, like "slow morning start to suit a relaxed pace", along with the profile fields it relates to. If the planner claims an item matches a preference the user never set, the checker rejects it.

### Reality Check

This is a set of plain Python rules, so the results are predictable and easy to test. It checks:

- **Dates**: the right number of days, and no days outside the trip.
- **Pace**: too many activities for the chosen pace, overlapping time slots, or an overloaded arrival day.
- **Budget**: activities plus the hotel (plus an airfare estimate, if the budget includes flights), plus a 10% buffer, compared with the budget. When it's over, the app works out concrete options: which items to drop, a cheaper stay, how many days would fit, or what the budget would need to be.
- **Weather**: outdoor activities on rainy or extreme-weather days that have no backup plan.
- **Sources**: citations to source IDs that don't exist, or prices marked as sourced without a source behind them.
- **Preferences**: meat dishes for a vegetarian, hikes for someone with a mobility need, or anything on the avoid list.

If something blocking comes up, the plan goes back to the planner once, along with the list of problems. Anything still wrong after that is shown to the user instead of being hidden.

The app also shows a row of scores:

| Score | What it measures |
|---|---|
| Budget | Whether estimated costs plus the buffer fit the budget |
| Pace | Whether every day stays within the pace limit, with no overlaps |
| Weather | Whether risky days have backup plans (n/a when the trip is beyond the forecast range) |
| Sourced | Share of stays and activities backed by a retrieved source |
| DNA fit | Share of activities that explain how they match the profile, minus conflicts |
| Overall | Average of the scores above |

Each score has a tooltip in the app with the same explanation.

### Rain backup plans

If the forecast for a day shows a 60% or higher chance of rain, or extreme heat or cold, every outdoor activity that day needs an indoor alternative. A plan without one gets sent back by the checker.

### Sources and data status

Every piece of external information keeps its provider, link, retrieval time and the date it applies to; these are listed in the Sources tab. Costs are labelled as estimates unless a source states them. Each data source shows whether it returned results, came back empty, was unavailable (with the reason), or failed. Fares, availability, bookings and forecasts are never made up.

### Feedback buttons

Under each plan there are buttons for "Too busy", "Too relaxed" and "Too expensive". They apply to the whole plan or to a single day. "Too busy" and "Too relaxed" adjust the pace; "Too expensive" asks for costs to come down by at least 25%. The plan is regenerated from the data already fetched, so no extra API calls are made. The change can also turn into a profile suggestion, which the user can accept or not. A "Fits me" button offers to save that trip's specific preferences to the profile.

### Flights

What's available depends on the AviationStack plan:

- **Free plan (default):** future schedules aren't available, so my MCP server has a tool, `get_route_flights_today`, that lists the airlines and departure times operating on the route today (for example, "IndiGo 6E 5321, dep 06:05"). It's labelled as a reference for which flights usually run, not a schedule for the travel date. If a flight number is mentioned ("I'm flying AI 657"), the app looks up its live status.
- **Basic plan (paid):** real schedules for the actual travel dates. Set `AVIATIONSTACK_PLAN=basic` in `.env`.

None of AviationStack's plans provide fares, so ticket prices are never shown. The free plan allows 100 requests a month, and results are reused within a conversation to save them.

### Weather

The weather tools geocode the destination and fetch the current conditions plus a daily forecast for the trip dates. OpenWeatherMap's free forecast only covers about 5 days ahead. Dates further out are marked as not covered, and the plan relies on web results about typical weather for that month.

### Error handling

- If one service is down, the rest of the app keeps working and the missing part is reported.
- If PostgreSQL is unreachable, the app falls back to in-memory storage and shows a warning that nothing will be saved.
- If Groq's free-tier rate limit is hit, the app waits and retries, up to twice, and says so on screen.
- Each user's data is stored under their own key.
- API keys live only in `.env`, which is excluded from git.

## Requirements

| What | Required | Where to get it | `.env` variable |
|---|---|---|---|
| Python 3.10+ | Yes | python.org | |
| Groq API key (free) | Yes | https://console.groq.com/keys | `GROQ_API_KEY` |
| PostgreSQL | Recommended | https://www.postgresql.org/download/ | `DATABASE_URL` |
| Tavily API key (free) | Recommended | https://app.tavily.com | `TAVILY_API_KEY` |
| OpenWeatherMap API key (free, takes 1-2 hours to activate) | Recommended | https://home.openweathermap.org/api_keys | `OPENWEATHER_API_KEY` |
| AviationStack API key (free) | Optional | https://aviationstack.com/signup/free | `AVIATIONSTACK_API_KEY` |
| uv (`pip install uv`) | For flight-number lookups | | |

For PostgreSQL, create an empty database called `travel_assistant` (pgAdmin works fine for this). The app creates its tables on first run.

## Setup

These commands are for Windows. On macOS or Linux, use `source .venv/bin/activate` and `cp` instead.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install uv
copy .env.example .env
```

Add the API keys to `.env`. The database URL looks like this:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/travel_assistant
```

Then:

```bash
python travel_assistant.py check        # checks keys, database and all three MCP servers
streamlit run travel_assistant.py       # starts the app at http://localhost:8501
```

There are also two other commands:

- `python travel_assistant.py chat` runs the same assistant in the terminal.
- `python travel_assistant.py selftest` runs the test suite. `pytest travel_assistant.py` also works.

## Using the app

1. Copy the traveller key shown at the top of the sidebar. Pasting it back under "Use a saved key" later restores the profile and past trips.
2. Fill in the Travel DNA form and save it. Each field has a tooltip.
3. Describe a trip in the chat box, e.g. "4 days in Goa from Mumbai from 20 December 2026, 2 adults, 50,000 rupees excluding flights".
4. Look through the scores and the Plan, Reality Check, Sources and Memory tabs.
5. Use the feedback buttons or type a change, like "make it 3 days".
6. Click "New trip" to plan somewhere else. The profile carries over.

## Tests

There are 22 tests. They use a fake LLM and fake tools, so they don't need API keys or internet access. One test starts the real MCP server as a separate process and talks to it over the protocol, with a local stub standing in for the weather and flight APIs. Another runs against a real PostgreSQL database when `DATABASE_URL` is reachable.

They cover:

- profile reuse and trip-level overrides
- suggestions staying pending until accepted
- isolation between users
- clarifying questions (no API calls spent)
- rejecting past dates
- the revision loop and its one-revision limit
- rain backups
- partial failures
- follow-up memory and feedback reusing fetched data
- hotel costs in the budget
- the accessibility rule
- parsing of flight and search results
- the Groq rate-limit retry

## Scope and future work

The assistant focuses on the planning stage of a trip: turning a rough idea into a realistic, personalised itinerary with sources attached. Booking is left to dedicated platforms, so costs in the plan are planning estimates rather than live quotes.

Directions I'd like to take it next:

- Live hotel and fare pricing through a booking or travel-pricing API
- Route-aware scheduling with a maps API for travel times between stops
- Multi-city trips and automatic currency conversion
- Longer-range weather using climate data, beyond the current 5-day forecast
- Full schedules for future travel dates on AviationStack's paid tier (already supported via `AVIATIONSTACK_PLAN=basic`)
- Proper user accounts in place of the traveller key

## Acknowledgement

Flight status lookups use the open-source [aviationstack-mcp](https://github.com/Pradumnasaraf/aviationstack-mcp) server (MIT license).

