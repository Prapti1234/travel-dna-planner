# Travel DNA Planner — Memory-Augmented Multi-Agent Travel Assistant

You describe a trip in plain English ("4 days in Goa from Mumbai from 20 Dec, 2 adults, ₹50,000"). A team of AI agents then:
- collects live data on flights, weather, hotels and things to do,
- writes a day-by-day plan tailored to **your saved travel personality**, and
- **checks that the plan is realistic** before showing it.

It also explains *why* each suggestion fits you, and *where* each fact came from.

**Stack:** Python · LangGraph · MCP · PostgreSQL · Groq · Tavily · AviationStack · OpenWeatherMap · Streamlit.
**Code:** one file, `travel_assistant.py`, with 22 built-in tests.

---

## 1. What problem does it solve?

A plain chatbot asked to plan a trip has three problems:
1. **It invents things:** prices, flight times, weather.
2. **It forgets you:** you repeat "I'm vegetarian, I like a slow pace" every time.
3. **It never checks itself:** 6 activities on a "relaxed" day, a beach walk in a storm, or a plan that blows the budget.

This project fixes all three: live data through tools, long-term memory in a database, and an automatic checker.

---

## 2. How it works, step by step

```
User request (Streamlit)
        |
        v
1. INTAKE agent (LLM)
     - turns the text into structured trip details (city, dates, budget, travellers)
     - loads the saved Travel DNA and this conversation's earlier turns (PostgreSQL)
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

**Key technologies, in one line each:**
- **LangGraph** is the "manager" connecting the agents. It holds shared *state* (the trip, the results, the plan), runs *nodes* (agents), and follows *edges* (who runs next, including the parallel step and the one-time loop back).
- **MCP (Model Context Protocol)** is a standard "plug" for tools. Each data source runs as a separate *MCP server* that advertises its tools; our app is the *MCP client* that discovers and calls them. Three servers are used:
  - **AviationStack server:** open-source, launched with `uvx`, runs on your PC
  - **our own travel-tools server:** inside `travel_assistant.py`, runs on your PC
  - **Tavily:** hosted online
- **PostgreSQL** is long-term memory. LangGraph's *checkpointer* saves every conversation, and LangGraph's *store* saves your Travel DNA.
- **Groq** runs the LLM (`openai/gpt-oss-120b` by default) very fast. It's used twice per plan, or three times if a revision is needed.
- **Streamlit** provides the web interface.

---

## 3. Every feature explained

### 1. Travel DNA (your saved travel personality)
**What:** a profile in the sidebar with these fields:
- travel style
- pace (relaxed = 3 activities a day, balanced = 4, packed = 6)
- accommodation type
- food
- mobility needs
- getting around
- favourite activities
- usual budget
- things to avoid

**How it's used:** it's saved in PostgreSQL and loaded automatically for **every** new trip, so you never repeat yourself. It shapes the web searches, the planner's choices and the Reality Check.

**Where to see it:** **Memory & assumptions** tab → "saved Travel DNA".

**Rule:** what you say for a *specific trip* overrides your DNA for that trip only. The app tells you about the override, and your DNA stays unchanged.

### 2. Memory you control (suggestions, not silent saving)
If you write "I'm vegetarian" or "I always prefer homestays" in chat, the app **does not** silently save it. It creates a **suggestion** in the sidebar, which you **Accept** or **Dismiss**. You can also:
- delete any saved field (the x button next to it)
- clear everything
- turn off "Suggest DNA updates from my messages"

**Why it matters:** the AI never turns a guess into a permanent fact about you.

### 3. Conversation memory (follow-ups)
Each trip is a conversation saved in PostgreSQL. Type "make it 3 days" or "add a beach day" and it keeps everything else (city, budget, people).

**Two kinds of memory:**
- **Checkpoint:** *this* trip's conversation.
- **Travel DNA:** *you*, across all trips.

Old trips appear under **Trips** in the sidebar, and they survive restarting the app.

### 4. "Why it fits you" explanations
Every activity and stay carries a short reason ("slow morning start matches your relaxed pace") and names the DNA fields it serves. The Reality Check **rejects** a plan that claims to match a preference you don't have.

### 5. Reality Check (the plan checker)
Plain Python rules, so the results are predictable and testable:

| Check | Example problem it catches |
|---|---|
| Dates | 3 plan days for a 4-day trip; dates outside your trip |
| Pace | 5 activities on a "relaxed" day; overlapping times; arrival day overloaded |
| Budget | Activities + **hotel** (+ estimated airfare) + 10% contingency over your budget. It then **calculates** alternatives: drop these items, choose a cheaper stay, shorten to N days, or raise the budget to X. |
| Weather | An outdoor activity on a rainy or extreme day with no Plan B |
| Sources | Citing a source ID that doesn't exist; a "verified" price with no source |
| Preferences | Meat dishes for a vegetarian; hikes for someone with a mobility need; anything on your "avoid" list |

If there's a blocking problem, the plan goes back to the planner **once** with the problem list. Anything still unresolved is shown openly, not hidden.

**The score row:**

| Score | Meaning |
|---|---|
| **Budget** | 100 = fits your budget |
| **Pace** | 100 = every day within your pace limit, no overlaps |
| **Weather** | 100 = all risky days have a Plan B; n/a when the dates are beyond the forecast |
| **Sourced** | % of the plan backed by real retrieved sources |
| **DNA fit** | % of activities that explain how they suit you, minus conflicts |
| **Overall** | The average of the scores above |

Hover the **?** next to each score in the app for the same explanation.

### 6. Rain Plan B
When the forecast shows a ≥60% chance of rain, or extreme heat or cold, every outdoor activity that day must come with an indoor alternative. If one is missing, the Reality Check sends the plan back.

### 7. Honest, evidence-aware answers
Every external fact keeps its **provider, link, retrieval time and the date it applies to** (see the **Sources** tab). Costs are marked **≈ estimate** unless a source states them. Each data source reports its real status:
- ok
- empty
- unavailable, with the reason
- error

The app never invents fares, availability, bookings or forecasts.

### 8. Feedback buttons
| Button | Effect |
|---|---|
| **Too busy** | Lowers the pace, for the whole plan or one chosen day |
| **Too relaxed** | Raises the pace |
| **Too expensive** | Cuts costs by at least 25% |

The plan is redone **using the data already collected**, so no new API calls are made. The change can also become a *suggestion* for your DNA, which you approve or not. **Fits me** offers to save this trip's special preferences to your DNA.

### 9. Flights: what works on which plan
| Your AviationStack plan | What the app does |
|---|---|
| **Free** (default) | Uses our MCP tool `get_route_flights_today`. It shows **which airlines fly your route today and at what times**, e.g. "IndiGo 6E 5321 dep 06:05", clearly labelled *"seen today: a guide, not a schedule for your date"*. It also checks the live status of any flight number you mention ("I'm flying AI 657"). |
| **Basic** (paid) | Real schedules for your actual travel dates (`AVIATIONSTACK_PLAN=basic` in `.env`). |

No AviationStack plan provides **fares**, so the app never shows ticket prices. The free plan allows 100 requests a month; the app reuses results within a conversation to save them.

### 10. Weather
Our own MCP server looks up the city's location, the current weather, and a day-by-day forecast for your trip dates. OpenWeatherMap's free forecast covers about **5 days ahead**. Later dates are marked "not covered" (nothing is invented), and the plan uses web sources about typical weather that month.

### 11. Reliability and safety
- **A service is down?** The rest still works, and you're told what's missing.
- **The database is down?** The app runs with temporary memory and shows a warning banner.
- **Groq's free limit is hit?** The app waits and retries automatically (up to twice) and tells you.
- **Separate users:** each person's data is stored under their own traveller key.
- **Secrets:** API keys live only in `.env`, which is never uploaded to GitHub.

---

## 4. What you need

| What | Required? | Where | `.env` name |
|---|---|---|---|
| Python **3.10+** | | python.org | — |
| Groq API key (free) | | https://console.groq.com/keys | `GROQ_API_KEY` |
| PostgreSQL | Recommended | https://www.postgresql.org/download/ (Windows installer, then create a database named `travel_assistant` in pgAdmin) | `DATABASE_URL` |
| Tavily API key (free) | Recommended | https://app.tavily.com | `TAVILY_API_KEY` |
| OpenWeatherMap API key (free; takes 1–2 h to activate) | Recommended | https://home.openweathermap.org/api_keys | `OPENWEATHER_API_KEY` |
| AviationStack API key (free) | Optional | https://aviationstack.com/signup/free | `AVIATIONSTACK_API_KEY` |
| uv (`pip install uv`) | For flight-number lookups | — | — |

## 5. Setup and run (Windows, VS Code terminal)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install uv
copy .env.example .env          # then fill in your keys (macOS/Linux: cp)

python travel_assistant.py check        # tests keys, database and all 3 MCP servers
streamlit run travel_assistant.py       # opens the app at http://localhost:8501
```

The `DATABASE_URL` format is `postgresql://postgres:YOUR_PASSWORD@localhost:5432/travel_assistant`. The app creates its tables automatically.

Other commands:
- `python travel_assistant.py chat`: the same assistant in the terminal.
- `python travel_assistant.py selftest`: 22 offline tests (`pytest travel_assistant.py` works too).

## 6. How to use the app

1. **Save your traveller key** (top of the sidebar). Paste it under *Use a saved key* to return later.
2. **Fill in your Travel DNA** and click **Save**. Hover the **?** icons for help on each field.
3. **Type a trip** in the chat box, e.g. *"4 days in Goa from Mumbai from 20 December 2026, 2 adults, ₹50,000 excluding flights"*.
4. **Read the result:** the score row, then the **Plan**, **Reality Check**, **Sources** and **Memory & assumptions** tabs.
5. **Adjust:** use the feedback buttons, or type a change ("make it 3 days").
6. **New trip:** click **New trip** in the sidebar. Your DNA is applied automatically.

## 7. Testing

The 22 automated tests use a fake AI and fake tools, so they need no keys and no internet. One test starts **our real MCP server** as a separate program and talks to it over MCP, with a local stand-in for the weather and flight APIs. Another test uses **real PostgreSQL** when `DATABASE_URL` is reachable.

They cover:
- Travel DNA reuse and trip overrides
- suggestions staying pending until accepted
- user isolation
- clarifying questions (no API calls spent) and past dates
- revision fixing problems, and revision limited to once
- Rain Plan B
- partial failures
- follow-up memory, and feedback reusing data
- hotel cost in the budget
- the accessibility rule
- parsing of route flights and Tavily results
- the Groq rate-limit retry

## 8. Known limitations

- No bookings, fares, or confirmed hotel prices or availability. Costs marked ≈ are AI estimates.
- Travel times between places are the AI's estimates; there's no maps API.
- Forecasts only cover ~5 days ahead.
- On the free AviationStack plan, flights are "seen today" patterns, not schedules for your date.
- Groq's free tier limits tokens per minute. Heavy use may need a short wait, and the app handles this automatically.
- No currency conversion. One base city per trip, up to 14 days.
- The traveller key is like a password, but there's no real login system.

## 9. Acknowledgements

The concept was inspired by Aarohi's public LangGraph + MCP travel-planner tutorials ([part 1](https://github.com/codewithaarohi/AI-Travel-Planning-System-using-LangGraph), [part 2](https://github.com/codewithaarohi/AI-Travel-Planning-App-using-LangGraph-and-MCP)). This is an independent implementation. Travel DNA, consent-based memory, explanations, the Reality Check, Rain Plan B, evidence tracking, feedback and the free-plan flight alternative are original additions.

Flight-status lookups use the open-source [aviationstack-mcp](https://github.com/Pradumnasaraf/aviationstack-mcp) server (MIT).
