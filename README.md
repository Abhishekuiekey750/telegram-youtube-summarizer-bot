# Telegram YouTube Summarizer Bot

A Telegram bot that summarizes YouTube videos and answers follow-up questions from the transcript. Send a YouTube link to get key points with timestamps, a core insight, and optional translation (e.g. Hindi). Then ask questions about the video and get answers grounded in the transcript.

---

## Setup

### Prerequisites

- **Python 3.10+**
- **Telegram Bot Token** — from [@BotFather](https://t.me/BotFather)
- **YouTube Data API key** — from [Google Cloud Console](https://console.cloud.google.com/) (APIs & Services → Credentials)
- **LLM API** — one of:
  - **OpenRouter** (recommended): [openrouter.ai/keys](https://openrouter.ai/keys) → set `OPENROUTER_API_KEY`
  - **OpenAI**: `OPENAI_API_KEY`
  - **Google Gemini**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → `GEMINI_API_KEY`

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/telegram-youtube-bot.git
   cd telegram-youtube-bot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`
   - Fill in your keys (see `.env.example` for all options):
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_MODE=polling
   YOUTUBE_API_KEY=your_youtube_api_key
   OPENROUTER_API_KEY=sk-or-...   # or OPENAI_API_KEY / GEMINI_API_KEY
   ```

5. **Run the bot**
   ```bash
   python -m cmd.bot.main
   ```

6. **Use in Telegram**
   - Open your bot and send `/start` or paste a YouTube link (e.g. `https://youtu.be/...`).
   - Use `/language` to choose summary language (e.g. Hindi).
   - After a summary, use **Ask Question** to query the transcript.

---

## Architecture

### High-level flow

```
Telegram Update → Dispatcher → Middleware (auth, rate limit, logging)
       → Classifier (command / youtube_link / question)
       → Handler (Command / Link / Question)
       → Services (YouTube, Summarizer, QA, Language)
       → Storage (session, vector stub, cache)
       → LLM (OpenRouter / OpenAI / Gemini)
```

### Main components

| Layer | Role |
|-------|------|
| **Entry** | `cmd/bot/main.py` — DI container, wiring, polling/webhook |
| **Bot** | `internal/bot/` — client, dispatcher, middleware, handlers, keyboards |
| **Handlers** | Command (`/start`, `/help`, `/language`), Link (URL → transcript → summary), Question (Q&A over last video) |
| **Services** | YouTube (metadata + transcript), Summarizer (chunking + LLM), QA (retriever + LLM), Language (translation) |
| **AI** | `internal/ai/` — model factory (OpenRouter/OpenAI/Gemini), embedding stub, prompt manager |
| **Storage** | Session (in-memory/Redis), vector DB (in-memory stub for chunk storage), cache |

### Summarization pipeline

1. **Transcript** — Fetched via `youtube-transcript-api`; validated with YouTube Data API (duration, title).
2. **Chunking** — Strategy by length: single chunk (short), overlapping segments (medium), or hierarchical (long). Chunks keep timestamps for key-point attribution.
3. **LLM** — Per-chunk or single-chunk prompt; response parsed for key points + core takeaway; missing timestamps filled from chunk time ranges.
4. **Merge** — For multi-chunk, optional merge prompt; if no templates, key points are concatenated and core insight taken from last chunk summary or first key point.
5. **Language** — If user chose non-English (e.g. Hindi), summary (and Q&A) can be translated via configured provider or free Google Translate fallback.

### Q&A pipeline

1. **Preparation** — When a video is summarized, the transcript is chunked (semantic chunker), and chunks are stored in the vector store (stub: in-memory by `video_id`).
2. **Retrieval** — On question, chunks for the current video are loaded; hybrid (keyword + optional vector) search returns relevant chunks.
3. **Answer** — LLM is prompted with question + retrieved context (+ optional conversation history); response is returned with grounding info.

### Project layout (relevant paths)

```
telegram-youtube-bot/
├── cmd/bot/main.py          # Entry, DI, run loop
├── config/                  # Config loader, base config
├── internal/
│   ├── ai/                  # Models, embeddings, prompts
│   ├── bot/                 # Client, dispatcher, handlers, middleware
│   ├── domain/              # Value objects, events
│   ├── pkg/                 # Logger, metrics, errors, retry
│   ├── services/            # YouTube, summarizer, QA, language
│   └── storage/             # Session, vector stub, cache
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design trade-offs

| Decision | Trade-off |
|----------|-----------|
| **In-memory vector store** | Simple, no infra; chunks are lost on restart. Acceptable for a single-instance bot; swap in Redis/Postgres + vector later for persistence. |
| **In-memory session store** | Same as above: no Redis required for setup; multi-instance would need a shared session store. |
| **Multiple LLM backends (OpenRouter, OpenAI, Gemini)** | One code path via `ModelFactory`; more branches and env keys in exchange for flexibility and cost/availability options. |
| **Chunking strategy by video length** | Short videos get one chunk (fast, one LLM call); long videos get multiple chunks and merge. Merge without templates uses heuristics (e.g. last chunk summary as core insight). |
| **Optional prompt templates** | Bot works with zero templates (built-in prompts); templates under `internal/ai/prompts/templates` allow customization and i18n without code changes. |
| **Translation** | Primary provider (e.g. OpenAI) can be complemented by free Google Translate so Hindi/other languages work without extra paid APIs. |
| **Polling vs webhook** | Default polling for easy local/dev use; webhook needs a public URL and TLS for production scale. |

---



---

## Demo video (3–5 minutes)

##A short demo video Link:
##https://drive.google.com/file/d/1lnm8A1LxRTPFm4ycxXIzoeQ8sCj2xQPP/view?usp=sharing


---


