# AI Startup Team

A full AI-powered startup crew with 7 agents that communicate via **Slack**, manage work in **Notion**, and are powered by **Google Gemini** through **CrewAI**.

## Team

| Role | What They Do |
|------|-------------|
| CEO | Strategy, vision, OKRs, coordinates the team, weekly synthesis |
| CFO | Financial models, burn rate, runway, fundraising strategy |
| VP of Product | PRDs, backlog prioritization, user stories, roadmap |
| Lead Developer | Architecture, tech stack, code, CI/CD, engineering execution |
| QA Lead | Test strategy, test plans, bug tracking, quality gates |
| Head of Marketing | GTM strategy, content, SEO, launch campaigns, brand |
| Head of Sales | Pipeline, outreach, pitch decks, objection handling, CRM |

## Architecture

```
main.py                          # Entry point (CLI + persistent mode)
config/settings.py               # Environment config (Gemini, Slack, Notion)
agents/factory.py                # Builds all 7 agents with tools
tools/slack_tools.py             # Send/read/reply Slack messages
tools/notion_tools.py            # Create pages, databases, tasks, query
workflows/slack_bot.py           # Real-time Slack listener + message router
workflows/scheduled_crews.py     # Standup, weekly review, sprint planning, kickoff
workflows/scheduler.py           # APScheduler for recurring team rituals
```

## Setup

### 1. Prerequisites

- Python 3.11+
- A Gemini API key ([get one free](https://aistudio.google.com/app/apikey))
- A Slack workspace with a bot app (Socket Mode enabled)
- A Notion integration with access to a root page

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual keys and IDs
```

### 4. Slack App Setup

Your Slack app needs these configurations:

**OAuth Scopes (Bot Token):**
- `app_mentions:read` — Detect @mentions
- `channels:history` — Read channel messages
- `channels:read` — List channels
- `chat:write` — Send messages
- `commands` — Slash commands
- `im:history` — Read DMs
- `im:read` — Detect DM events
- `im:write` — Send DMs

**Socket Mode:** Enable Socket Mode and generate an App-Level Token (`xapp-...`)

**Event Subscriptions:** Subscribe to `app_mention` and `message.im` events

**Slash Commands:** Create `/team` command pointed to your app

### 5. Notion Setup

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Create a root page in Notion for the AI team workspace
3. Share the root page with your integration
4. Copy the page ID from the URL into your `.env`

### 6. Slack Channels

Create these channels in Slack and add your bot:
- `#ai-team-general` — All-team communication
- `#ai-team-engineering` — Dev + QA discussions
- `#ai-team-business` — Marketing + Sales updates
- `#ai-team-executive` — CEO + CFO strategic decisions

Copy each channel's ID into your `.env`.

## Usage

### Full system (Slack bot + scheduled workflows)
```bash
python main.py
```

### Run the idea-to-plan kickoff
```bash
python main.py --kickoff
python main.py --kickoff --idea "Your custom startup idea here"
```

### One-off workflows
```bash
python main.py --standup       # Daily standup
python main.py --review        # Weekly strategy review
python main.py --sprint        # Sprint planning session
```

### Partial modes
```bash
python main.py --slack-only      # Slack bot without scheduler
python main.py --schedule-only   # Scheduler without Slack bot
```

### Slack interaction

Once running, interact with your team in Slack:

- **@mention the bot** in any channel with a request — it routes to the right agent
- **DM the bot** for private requests
- **`/team <request>`** slash command for quick routing

The bot automatically determines which agent should handle each message based on keywords (finance -> CFO, bugs -> QA, features -> Product, etc.).

## Scheduled Workflows

| Workflow | Schedule | What Happens |
|----------|----------|-------------|
| Daily Standup | Mon-Fri 9:00 AM | Each agent reports progress, blockers; CEO synthesizes |
| Weekly Review | Friday 4:00 PM | Full team status reports; CEO writes weekly summary |
| Sprint Planning | Every other Monday 10:00 AM | Product scopes, Dev estimates, QA plans, CEO approves |

## Notion Databases

The agents automatically create and manage these Notion databases:

- **Tasks** — Kanban board with status, priority, assignee, due dates
- **Sprints** — Sprint records with goals and dates
- **Leads** — Sales pipeline with stages and deal values
- **Bugs** — Bug tracker with severity and reproduction steps

## Customizing

- **Change LLM model** — Edit `GEMINI_MODEL` in `.env` (e.g., `gemini-2.0-pro`)
- **Adjust schedules** — Modify cron triggers in `workflows/scheduler.py`
- **Add new agents** — Add a spec to `_AGENT_SPECS` in `agents/factory.py`
- **Add custom tools** — Create tools in `tools/` following the CrewAI `BaseTool` pattern
- **Switch LLM provider** — Replace `ChatGoogleGenerativeAI` with `ChatOpenAI` in `agents/factory.py`
