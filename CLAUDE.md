# Project Rules

## Python environment

**Always use the virtual environment in this project.**

- Activate: `source .venv/bin/activate`
- Python binary: `.venv/bin/python`
- Pip binary: `.venv/bin/pip`

Never run `python` or `pip` directly — always prefix with `.venv/bin/` or activate first.

```bash
# Running scripts
.venv/bin/python main.py
.venv/bin/python setup_slack_channel.py

# Installing packages
.venv/bin/pip install -r requirements.txt

# Adding a new package
.venv/bin/pip install some-package
# Then pin it: .venv/bin/pip freeze | grep some-package >> requirements.txt
```

## First-time setup

Package installation must be run locally (not in the Claude sandbox — no outbound pip access there).

```bash
# In your terminal, from the project folder:
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:
1. Create `.venv` using your local Python 3
2. `pip install -r requirements.txt` inside it
3. Check `.env` for missing credentials
4. Run `setup_slack_channel.py` to create `#ai-team-standup`

## Environment variables

All secrets live in `.env`. Never hardcode credentials.
Copy `.env.example` → `.env` and fill in all values before running anything.

Required keys:
- `GEMINI_API_KEY` — from https://aistudio.google.com/app/apikey
- `SLACK_BOT_TOKEN` — `xoxb-...` from your Slack app
- `SLACK_APP_TOKEN` — `xapp-...` (Socket Mode token)
- `SLACK_SIGNING_SECRET` — from your Slack app Basic Information page
- `NOTION_API_KEY` — from https://www.notion.so/my-integrations
- `NOTION_ROOT_PAGE_ID` — ID of the Notion page shared with your integration

## Project layout

```
agents/                  Agent role definitions (7 roles)
config/                  Settings loaded from .env
tools/                   Slack + Notion tool wrappers
workflows/               Scheduled crews, Slack bot, scheduler
main.py                  Entry point
setup.sh                 First-time local setup script
setup_slack_channel.py   One-time Slack channel creator
```
