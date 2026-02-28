# Demo Video Script (3–5 minutes)

Use this as a loose script for recording your submission demo.

---

## 1. Intro (≈30 s)

- “This is the Telegram YouTube Summarizer Bot.”
- “You send a YouTube link, get a summary with key points and timestamps, and can ask questions about the video.”
- “Summaries can be in English or other languages like Hindi.”

---

## 2. Setup (≈30 s, optional)

- Show project root: `README`, `requirements.txt`, `.env.example`.
- “Copy `.env.example` to `.env`, add your Telegram token, YouTube API key, and an LLM key — OpenRouter, OpenAI, or Gemini.”
- Run: `python -m cmd.bot.main`, show “application.started” or “polling_started” in the terminal.

---

## 3. Main flow (2–3 min)

- **Start:** Open the bot in Telegram, send `/start`, briefly show welcome and `/help`.
- **Link:** Paste a YouTube URL (e.g. a short talk). Show “Fetching transcript…” / progress, then the summary message.
- **Summary:** Point out key points with timestamps and the core insight.
- **Q&A:** Tap “Ask Question”, type e.g. “What did he say about [topic]?”, show the answer.
- **Language (optional):** Send `/language`, choose Hindi (or another), send the same or another link, show summary in that language.

---

## 4. Architecture (≈1 min, optional)

- Open `README.md`, scroll to “Architecture” and “Design trade-offs”.
- One sentence each: “Updates go through a dispatcher and middleware, then handlers call services for YouTube, summarization, and Q&A; we use an in-memory vector store and support several LLM backends.”

---

## 5. Outro (≈15 s)

- “Repo and setup are in the README; screenshots are in the README too. Thanks.”

---

**Tips:** Record in one take or cut per section. Keep the bot and one Telegram window visible. If you show the code, zoom so filenames and key lines are readable.
