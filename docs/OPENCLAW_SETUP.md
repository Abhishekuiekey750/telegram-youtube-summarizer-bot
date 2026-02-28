# OpenClaw as the Intelligent Assistant Layer

This document describes how to **install OpenClaw locally** and use it as the **LLM backend** for this Telegram YouTube Summarizer bot, satisfying the assignment’s “full pipeline” and “connect Telegram with OpenClaw skill” in an integrated way.

## Architecture

- **This repo**: Telegram bot that receives YouTube links, fetches transcripts, and sends summary/Q&A requests to an LLM.
- **OpenClaw**: When configured, acts as the **intelligent assistant layer** — all summarization and Q&A generation goes through the OpenClaw gateway instead of calling OpenAI (or a stub) directly.
- **Pipeline**: Telegram → Python bot → YouTube transcript → **OpenClaw gateway** (LLM) → summary → Telegram.

So you are not “just calling an API”; you are **architecting a pipeline** where OpenClaw is the local AI layer.

---

## 1. Install OpenClaw locally

**Requirements**: Node.js **22+** (required by OpenClaw).

1. Install Node 22+ from [nodejs.org](https://nodejs.org/) or via `nvm`/`fnm`.
2. Install the OpenClaw CLI and run the gateway:

   ```bash
   # Option A: npm (global)
   npm install -g openclaw

   # Option B: pnpm
   pnpm add -g openclaw

   # Option C: run via npx without global install
   npx openclaw gateway
   ```

   On **Windows**, the recommended way is **WSL2** (see [OpenClaw Windows (WSL2)](https://docs.openclaw.ai/platforms/windows.md)). Install Node 22 inside WSL, then run the commands above there.

3. Confirm it runs:

   ```bash
   openclaw --version
   openclaw gateway
   ```

   By default the gateway listens on a port (e.g. `18789`). Note the port and the base URL: `http://127.0.0.1:18789`.

---

## 2. Enable the OpenAI-compatible Chat Completions endpoint

The bot talks to OpenClaw via its **OpenAI-compatible HTTP API**. You must enable that endpoint in the OpenClaw gateway config.

1. Find or create the config file. Common locations:
   - `~/.openclaw/openclaw.json` (Linux/macOS)
   - `%USERPROFILE%\.openclaw\openclaw.json` (Windows)
   - Or path given in [Configuration](https://docs.openclaw.ai/gateway/configuration.md).

2. Enable the chat completions endpoint and (if needed) set auth. Example snippet:

   ```json5
   {
     gateway: {
       http: {
         endpoints: {
           chatCompletions: { enabled: true },
         },
       },
       auth: {
         mode: "token",
         token: "YOUR_GATEWAY_TOKEN",
       },
     },
     channels: {
       telegram: {
         enabled: true,
         botToken: "YOUR_TELEGRAM_BOT_TOKEN",
         dmPolicy: "pairing",
       },
     },
   }
   ```

   Replace `YOUR_GATEWAY_TOKEN` with a secret you choose (the bot will send this as `OPENAI_API_KEY`). Replace `YOUR_TELEGRAM_BOT_TOKEN` if you want OpenClaw’s **native** Telegram channel; for **this repo’s bot** you only need the gateway token and the endpoint enabled.

   See [OpenClaw OpenAI HTTP API](https://docs.openclaw.ai/gateway/openai-http-api.md) and [Configuration Reference](https://docs.openclaw.ai/gateway/configuration-reference.md).

3. Restart the gateway:

   ```bash
   openclaw gateway
   ```

---

## 3. Connect this bot to the OpenClaw gateway

1. In this project’s **`.env`** (not `.env.example`), set:

   ```env
   OPENAI_BASE_URL=http://127.0.0.1:18789/v1
   OPENAI_API_KEY=YOUR_GATEWAY_TOKEN
   ```

   Use the **same** `YOUR_GATEWAY_TOKEN` you set in `gateway.auth.token` in OpenClaw config.  
   Use the correct port if your gateway uses something other than `18789` (e.g. `http://127.0.0.1:PORT/v1`).

2. Optional: use a specific OpenClaw agent:

   ```env
   OPENCLAW_AGENT_ID=main
   ```

   Default is `main`; the bot will send `model: "openclaw:main"` (or `openclaw:<OPENCLAW_AGENT_ID>`).

3. Start the **Telegram YouTube Summarizer** bot:

   ```bash
   cd D:\Projects\telegram-youtube-bot
   .\venv\Scripts\activate
   python -m cmd.bot.main
   ```

   You should see a log line like:  
   `model_factory.openai_enabled` with `base_url=http://127.0.0.1:18789/v1`.

4. In Telegram, send a YouTube link to **your bot**. The bot will:
   - Fetch the transcript,
   - Send the summarization request to **OpenClaw** (your local gateway),
   - Reply with the summary.

That’s the full pipeline: **Telegram → this bot → OpenClaw (intelligent assistant layer) → summary**.

---

## 4. Connect Telegram with OpenClaw (native channel, optional)

The assignment may also ask to “connect Telegram with OpenClaw skill”. You can do that in two ways:

- **A. This repo’s bot (recommended for the assignment)**  
  The bot in this repo is already “connected” to OpenClaw: it receives Telegram messages and uses OpenClaw as the LLM. No extra step.

- **B. OpenClaw’s own Telegram channel**  
  To also run OpenClaw’s native Telegram interface (separate from this bot):
  1. In OpenClaw config, set `channels.telegram.enabled: true` and `channels.telegram.botToken` (can be the same or a different bot).
  2. Run `openclaw gateway` and use pairing:
     ```bash
     openclaw pairing list telegram
     openclaw pairing approve telegram <CODE>
     ```
  3. Then you have two things: (1) this repo’s YouTube summarizer bot, and (2) OpenClaw’s generic Telegram bot. Both can coexist.

For “use it as the intelligent assistant layer”, **A** is enough: this bot + OpenClaw gateway.

---

## 5. If you don’t use OpenClaw (direct OpenAI)

To use **OpenAI** directly (no OpenClaw):

1. Do **not** set `OPENAI_BASE_URL`.
2. In `.env` set only:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   ```
3. Run the bot as above. You should see `model_factory.openai_enabled` without a `base_url`. Summaries will come from OpenAI, not from a local OpenClaw gateway.

---

## 6. Troubleshooting

- **`model_factory.openai_disabled reason=missing_sdk_or_api_key`**  
  - Ensure `.env` has `OPENAI_API_KEY` set and is in the project root (or loaded by your shell).  
  - For OpenClaw, also set `OPENAI_BASE_URL=http://127.0.0.1:18789/v1` (correct port if different).

- **Bot starts but summaries are generic / “demo purposes”**  
  - Either the stub is still in use (no valid `OPENAI_API_KEY` / wrong key), or the gateway is not reachable.  
  - Check that `openclaw gateway` is running and that `gateway.http.endpoints.chatCompletions.enabled` is `true`.  
  - Test the endpoint:
    ```bash
    curl -sS http://127.0.0.1:18789/v1/chat/completions \
      -H "Authorization: Bearer YOUR_GATEWAY_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"model":"openclaw:main","messages":[{"role":"user","content":"Hi"}]}'
    ```

- **Windows**  
  Use WSL2 and install Node 22 + OpenClaw inside WSL; run the gateway there and point the Windows-side `.env` to `http://127.0.0.1:18789/v1` (localhost is shared).

---

## Summary

| Goal                         | What to do                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| Install OpenClaw locally    | Install Node 22+, then `npm install -g openclaw` (or pnpm / npx); use WSL2 on Windows. |
| Use OpenClaw as LLM layer   | Enable `chatCompletions` in gateway config; set `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `.env`. |
| Connect Telegram + OpenClaw | This bot already uses OpenClaw when `OPENAI_BASE_URL` is set; optionally add OpenClaw’s native Telegram channel. |
| Full pipeline               | Telegram → Python bot → YouTube transcript → OpenClaw gateway → summary → Telegram. |
