# AI Startup Team — Setup Checklist

## Required Before First Run

- [ ] Set your Gemini API key in `.env`
- [ ] Create and configure a Slack app (see README for required scopes)
- [ ] Set Slack tokens in `.env` (bot token, app token, signing secret)
- [ ] Create 4 Slack channels and add the bot to each
- [ ] Create a Notion integration and set the API key in `.env`
- [ ] Create a Notion root page, share it with the integration, and set the page ID in `.env`
- [ ] Install dependencies: `pip install -r requirements.txt`

## First Run

- [ ] Run `python main.py --kickoff` to generate the full startup plan
- [ ] Check Slack channels for agent messages
- [ ] Check Notion for generated pages and databases

## Optional Enhancements

- [ ] Customize agent personas in `agents/factory.py`
- [ ] Adjust scheduled times in `workflows/scheduler.py`
- [ ] Add web search tools (SerperDevTool) for real-time market research
- [ ] Add long-term memory with a vector store for cross-session context
- [ ] Deploy to a server for 24/7 operation (Docker, Railway, Fly.io)
