#!/usr/bin/env bash
# setup.sh — One-time local setup for the AI Startup Team
#
# Run from the project folder on your Mac:
#   chmod +x setup.sh && ./setup.sh

set -e

echo ""
echo "========================================================================"
echo "  AI Startup Team - Setup"
echo "========================================================================"

# Helper: returns 0 if the given Python binary is 3.10–3.13
python_version_ok() {
  "$1" -c "import sys; v=sys.version_info; exit(0 if (3,10)<=v<(3,14) else 1)" 2>/dev/null
}

python_version_str() {
  "$1" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))" 2>/dev/null
}

# ── Step 0: Find a compatible Python ────────────────────────────────────────
echo ""
echo "Step 0/4  Finding Python 3.10-3.13 (required by crewai)..."

PYTHON=""
for candidate in \
    python3.13 python3.12 python3.11 python3.10 \
    /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
    /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
    /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
    /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
    "$HOME/.pyenv/shims/python3" python3; do
  if command -v "$candidate" &>/dev/null 2>&1 || [ -x "$candidate" ]; then
    if python_version_ok "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  FOUND_VER=""
  for fb in python3 python; do
    command -v "$fb" &>/dev/null && FOUND_VER="$($fb --version 2>&1)" && break
  done

  echo ""
  echo "  ERROR: No compatible Python (3.10-3.13) found."
  [ -n "$FOUND_VER" ] && echo "  You have: $FOUND_VER — crewai does not support Python 3.14+ yet."
  echo ""
  echo "  Fix (pick one):"
  echo ""
  echo "  A) Homebrew (fastest):"
  echo "       brew install python@3.12"
  echo "       ./setup.sh"
  echo ""
  echo "  B) pyenv (version manager):"
  echo "       brew install pyenv"
  echo "       pyenv install 3.12 && pyenv local 3.12"
  echo "       ./setup.sh"
  echo ""
  echo "  C) python.org installer:"
  echo "       https://www.python.org/downloads/release/python-3120/"
  echo "       Install, then re-run ./setup.sh"
  echo ""
  exit 1
fi

echo "  Using: $PYTHON ($(python_version_str "$PYTHON"))"

# ── Step 1: Virtual environment ──────────────────────────────────────────────
echo ""
echo "Step 1/4  Setting up virtual environment (.venv)..."

REBUILD=false
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
  if python_version_ok ".venv/bin/python"; then
    echo "  .venv already OK (Python $(python_version_str .venv/bin/python)) — skipping"
  else
    echo "  .venv has wrong Python ($(python_version_str .venv/bin/python 2>/dev/null || echo unknown)) — rebuilding..."
    REBUILD=true
  fi
else
  REBUILD=true
fi

if [ "$REBUILD" = true ]; then
  rm -rf .venv
  "$PYTHON" -m venv .venv
  echo "  Created .venv with Python $(python_version_str .venv/bin/python)"
fi

# ── Step 2: Install requirements ─────────────────────────────────────────────
echo ""
echo "Step 2/4  Installing packages (may take 1-2 minutes)..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt
echo "  All packages installed."

# ── Step 3: .env check ───────────────────────────────────────────────────────
echo ""
echo "Step 3/4  Checking .env..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "  Created .env from .env.example"
  echo ""
  echo "  Fill in these values before continuing:"
  echo "    GEMINI_API_KEY       https://aistudio.google.com/app/apikey"
  echo "    SLACK_BOT_TOKEN      xoxb-... from api.slack.com"
  echo "    SLACK_APP_TOKEN      xapp-... (Socket Mode)"
  echo "    SLACK_SIGNING_SECRET from app Basic Information"
  echo "    NOTION_API_KEY       https://www.notion.so/my-integrations"
  echo "    NOTION_ROOT_PAGE_ID  page ID from your Notion URL"
  echo ""
  read -rp "  Press Enter once .env is filled in..."
else
  echo "  .env exists."
fi

if grep -qE "your-gemini-api-key-here|xoxb-your-bot-token|secret_your-notion|your-root-page-id|xapp-your-app-token" .env 2>/dev/null; then
  echo "  WARNING: placeholder values still in .env — please fill them in."
  read -rp "  Press Enter once done..."
fi

# ── Step 4: Slack channel ────────────────────────────────────────────────────
echo ""
echo "Step 4/4  Creating #ai-team-standup channel..."
.venv/bin/python setup_slack_channel.py

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  Setup complete!"
echo "========================================================================"
echo ""
echo "  Start the system:       .venv/bin/python main.py"
echo "  Run a kickoff:          .venv/bin/python main.py --kickoff"
echo "  Trigger standup now:    .venv/bin/python main.py --standup"
echo ""
