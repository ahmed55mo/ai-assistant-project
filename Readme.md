# AI Assistant & Automation System

## Architecture

`POST /api/chat` remains the only frontend chat interface. The backend stores
conversation messages in process memory and runs this pipeline:

```text
user -> ConversationService -> Groq tool selection -> ToolRegistry
     -> tool adapter -> structured tool result -> Groq -> final response
```

Tools are registered with a name, description, JSON input schema, executable
adapter, read-only flag, and confirmation requirement. The LLM can request
multiple tools in one response. The loop is capped by `MAX_TOOL_ITERATIONS`.

Conversation memory stores `system` implicitly through the LLM service and
`user`, `assistant`, and `tool` messages. OAuth credentials are never stored
in conversation memory.

## Tools

- `calculator`: fully functional local safe arithmetic evaluator.
- Gmail: real Gmail API read tools (`search_emails`, `get_email`,
  `get_recent_emails`, `summarize_emails`, and `identify_important_emails`).
- Calendar: provider abstraction with a safe mock provider by default and a
  Google Calendar provider when `CALENDAR_PROVIDER=google`.
- Notifications: Resend email SDK and Telegram Bot API adapters. Both require
  confirmation and real credentials; they never report success without a
  provider response.

Calendar mutations and notifications are side-effecting and require an
explicit confirmation message such as `yes`. A model response alone cannot
authorize them.

## Gmail and Google OAuth setup

1. In Google Cloud Console, create or select a project.
2. Enable **Gmail API**. Enable **Google Calendar API** if using the Google
   calendar provider.
3. Configure the OAuth consent screen for a testing/local application.
4. Create an OAuth client ID for a **Web application**.
5. Add this authorized redirect URI:
   `http://127.0.0.1:8000/auth/google/callback`
6. Download the client JSON to `credentials.json` in the project root. It is
   ignored by Git.
7. Start the backend and open `http://127.0.0.1:8000/auth/google`.
8. Grant the requested read Gmail and Calendar permissions. The backend stores
   the local refresh token in `token.json`, also ignored by Git.

The Gmail scope is read-only. Calendar uses the calendar scope because future
event creation/update/delete capabilities require it.

## Environment variables

Copy `backend/.env.example` values into the root `.env` and set:

- `GROQ_API_KEY`
- `MODEL_NAME`
- `LLM_TIMEOUT_SECONDS`
- `MAX_TOOL_ITERATIONS`
- `GOOGLE_CLIENT_SECRET_FILE`
- `GOOGLE_TOKEN_FILE`
- `GOOGLE_REDIRECT_URI`
- `CALENDAR_PROVIDER` (`mock` or `google`)
- `APP_TIMEZONE` (IANA timezone used when calendar requests omit a timezone)
- `RESEND_API_KEY`, `RESEND_FROM_EMAIL`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

Do not put credentials in frontend files.

## Run

From `D:\ai-assistant-automation`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The frontend can continue to be opened using the existing static-file
workflow. It still sends `{message, conversation_id}` and receives
`{conversation_id, response}`.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Automated tests use mocked Groq completions, Gmail API responses, a calendar
mock provider, and notification definitions. Real Gmail access requires the Google Cloud setup and browser OAuth flow above.
Real Calendar, Resend, and Telegram calls require their corresponding
credentials and are not executed by automated tests.

## Notifications

Email notifications use the official Resend Python SDK. Create a Resend API
key and verify the sender domain or address, then set `RESEND_API_KEY` and
`RESEND_FROM_EMAIL`. Sending always requires an explicit `yes` confirmation.
Telegram notifications use a bot token from BotFather and a target chat ID.
Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

Both `send_email` and `send_telegram_message` require an explicit `yes`
confirmation. Example prompts:

```text
Send me a Telegram message saying the assistant is working.
Send an email to me saying the project meeting is confirmed.
```