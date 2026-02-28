# Fix: "No API key found for provider anthropic" / OpenClaw chat completion failed

Your OpenClaw gateway is set to use **Anthropic (Claude)** (`anthropic/claude-opus-4-6`) but has **no API key** for that provider. When the bot sends a summary request, OpenClaw fails and you see "OpenClaw/API error" in Telegram.

---

## Option A: Add Anthropic (Claude) API key to OpenClaw

1. Get an API key from [Anthropic Console](https://console.anthropic.com/).
2. In PowerShell, set it and start the gateway:
   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   openclaw gateway
   ```
   Or run the OpenClaw auth wizard and add the key:
   ```bash
   openclaw agents add main
   ```
   (Follow the prompts to add Anthropic.)
3. Restart the gateway (Ctrl+C, then `openclaw gateway` again).

---

## Option B: Use OpenAI from OpenClaw instead of Claude

1. Edit `C:\Users\abhis\.openclaw\openclaw.json`.
2. Change the default agent model from `anthropic/claude-opus-4-6` to an OpenAI model (e.g. `openai/gpt-4o-mini`). See [OpenClaw config](https://docs.openclaw.ai/gateway/configuration) and [OpenAI provider](https://docs.openclaw.ai/providers/openai).
3. Set your OpenAI API key for OpenClaw (env `OPENAI_API_KEY` or in config).
4. Restart the gateway.

---

## Option C: Bypass OpenClaw – use OpenAI directly in the bot

If you only need summaries to work and don’t need OpenClaw for the assignment:

1. Open the bot’s **`.env`** in `D:\Projects\telegram-youtube-bot`.
2. **Remove** the line: `OPENAI_BASE_URL=http://127.0.0.1:18789/v1`
3. Set only: `OPENAI_API_KEY=sk-your-openai-api-key` (get one from [OpenAI](https://platform.openai.com/api-keys))
4. Restart the Python bot. You can stop the OpenClaw gateway. The bot will call OpenAI directly and summaries will work.
