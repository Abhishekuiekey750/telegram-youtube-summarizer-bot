"""
Command Handler
Processes all bot commands: /start, /help, /language
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import structlog

from internal.bot.handlers.base import BaseHandler
from internal.bot.dispatcher import UpdateContext
from internal.bot.keyboard import KeyboardBuilder
from internal.services.language import LanguageService
from internal.services.summarizer import SummarizerService, SummaryType
from internal.services.youtube import YouTubeService, Transcript
from internal.storage.session import SessionStore, UserSession
from internal.domain.value_objects import Language
from internal.pkg.errors import ValidationError, ErrorKind


class CommandHandler(BaseHandler):
    """
    Handles all bot commands with session integration.
    
    Commands supported:
    - /start - Initialize bot and show welcome
    - /help - Show help information
    - /language - Change user language preference
    
    Features:
    - Session management for user preferences
    - Multilingual responses
    - Interactive keyboards for language selection
    - Command analytics tracking
    """
    
    def __init__(
        self,
        telegram_client,
        session_store: SessionStore,
        language_service: LanguageService,
        summarizer_service: Optional[SummarizerService] = None,
        youtube_service: Optional[YouTubeService] = None,
        logger=None,
        metrics=None,
    ):
        """
        Initialize command handler with dependencies.

        Args:
            telegram_client: Client for sending messages
            session_store: Session manager for user data
            language_service: Service for multilingual support
            summarizer_service: For /deepdive (detailed summary)
            youtube_service: For /deepdive (fetch transcript)
            logger: Structured logger
            metrics: Metrics collector
        """
        super().__init__(telegram_client, logger, metrics)

        self.session_store = session_store
        self.language_service = language_service
        self.summarizer_service = summarizer_service
        self.youtube_service = youtube_service

        # Command registry
        self._commands: Dict[str, Dict[str, Any]] = {
            "start": {
                "handler": self._handle_start,
                "description": "Initialize the bot",
                "aliases": ["s", "begin"],
                "needs_session": False,
            },
            "help": {
                "handler": self._handle_help,
                "description": "Show help information",
                "aliases": ["h", "?"],
                "needs_session": False,
            },
            "language": {
                "handler": self._handle_language,
                "description": "Change language preference",
                "aliases": ["lang", "भाषा"],
                "needs_session": True,
            },
            "summary": {
                "handler": self._handle_summary,
                "description": "Show last video summary again",
                "aliases": [],
                "needs_session": True,
            },
            "deepdive": {
                "handler": self._handle_deepdive,
                "description": "Get a detailed summary of the current video",
                "aliases": ["deep", "detailed"],
                "needs_session": True,
            },
            "actionpoints": {
                "handler": self._handle_actionpoints,
                "description": "Show action items from the last video",
                "aliases": ["actions", "ap"],
                "needs_session": True,
            },
        }
        
        # Build command descriptions for help
        self._help_text = self._build_help_text()
        
        self.logger.info(
            "command_handler.initialized",
            commands=list(self._commands.keys()),
        )
    
    @property
    def handler_name(self) -> str:
        return "command"
    
    async def handle(self, context: UpdateContext) -> None:
        """
        Main entry point for command processing.
        
        Steps:
        1. Validate input
        2. Parse command
        3. Load user session
        4. Route to specific command handler
        5. Update session if needed
        6. Send response
        """
        start_time = datetime.now()

        try:
            await self.before_handle(context)

            # If this is a callback (e.g. language button), handle it and return
            if getattr(context.update, "callback_query", None):
                await self.handle_callback(context)
                return

            # Validate required fields
            await self._validate_required_fields(
                context,
                ["chat_id", "user_id", "message_text"],
            )
            
            # Parse command from message
            command, args = self._parse_command(context.message_text)
            
            if not command:
                await self._send_error_message(
                    context.chat_id,
                    "Invalid command format. Use /help to see available commands.",
                )
                return
            
            # Get command handler
            command_info = self._get_command(command)
            if not command_info:
                await self._handle_unknown_command(context, command)
                return
            
            # Load user session (if needed)
            session = None
            if command_info["needs_session"]:
                session = await self.session_store.get_session(context.user_id)
                if not session:
                    # Create new session if doesn't exist
                    session = await self.session_store.create_session(
                        user_id=context.user_id,
                        chat_id=context.chat_id,
                    )
                
                # Add language to context for response
                context.metadata["language"] = session.language
            
            # Execute command handler
            handler = command_info["handler"]
            response = await handler(context, session, args)
            
            # Update session if modified
            if session and session.modified:
                await self.session_store.save_session(session)
            
            # Send response if handler didn't send one
            if response and isinstance(response, str):
                await self._send_response(
                    chat_id=context.chat_id,
                    text=response,
                    parse_mode="Markdown",
                    reply_to_message_id=context.update.effective_message.message_id,
                )
            
            # Track command usage
            self._track_user_activity(
                context.user_id,
                f"command_{command}",
            )
            
            await self.after_handle(context)
            
        except Exception as e:
            await self.on_error(context, e)
            raise
            
        finally:
            self._track_processing_time(start_time)
    
    def _parse_command(self, text: str) -> tuple[Optional[str], str]:
        """
        Parse command from message text.
        
        Examples:
        "/start" → ("start", "")
        "/language hi" → ("language", "hi")
        "/lang@bot Tamil" → ("lang", "Tamil")
        
        Args:
            text: Message text
            
        Returns:
            Tuple of (command_name, arguments)
        """
        if not text or not text.startswith("/"):
            return None, ""
        
        # Remove leading slash and split
        parts = text[1:].split()
        if not parts:
            return None, ""
        
        command_part = parts[0].lower()
        
        # Remove bot username if present (e.g., /start@bot)
        if "@" in command_part:
            command_part = command_part.split("@")[0]
        
        # Get arguments (everything after command)
        args = " ".join(parts[1:]) if len(parts) > 1 else ""
        
        return command_part, args
    
    def _get_command(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Get command info by name or alias.
        
        Args:
            command: Command name or alias
            
        Returns:
            Command info dict or None if not found
        """
        # Check exact match
        if command in self._commands:
            return self._commands[command]
        
        # Check aliases
        for cmd_name, cmd_info in self._commands.items():
            if command in cmd_info.get("aliases", []):
                return cmd_info
        
        return None
    
    def _build_help_text(self) -> str:
        """
        Build help text from command registry.
        
        Returns:
            Formatted help text
        """
        lines = ["🤖 *YouTube Summarizer Bot Help*\n"]
        
        for cmd_name, cmd_info in self._commands.items():
            description = cmd_info["description"]
            aliases = cmd_info.get("aliases", [])
            
            cmd_line = f"/{cmd_name}"
            if aliases:
                alias_str = ", ".join(f"/{a}" for a in aliases[:2])
                cmd_line += f" ({alias_str})"
            
            lines.append(f"• {cmd_line} - {description}")
        
        lines.extend([
            "\n*How to use:*",
            "1. Send any YouTube link",
            "2. Get summary with key points, timestamps & core insight",
            "3. Use the *Ask Question* button or type your question about the video",
            "",
            "❓ *Q&A:* After a summary, tap *Ask Question* below the message or type your question (e.g. _What did he say about education?_).",
            "",
            "🌐 *Supported Languages:*",
            "English • हिन्दी • தமிழ் • తెలుగు • ಕನ್ನಡ",
        ])
        
        return "\n".join(lines)
    
    # ------------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------------
    
    async def _handle_start(
        self,
        context: UpdateContext,
        session: Optional[UserSession],
        args: str,
    ) -> Optional[str]:
        """
        Handle /start command.
        
        Creates user session and shows welcome message.
        
        Args:
            context: Update context
            session: User session (may be None)
            args: Command arguments
            
        Returns:
            Response text or None if response already sent
        """
        # Create session if doesn't exist
        if not session:
            session = await self.session_store.create_session(
                user_id=context.user_id,
                chat_id=context.chat_id,
            )
        
        # Determine language for welcome
        language = session.language
        
        # Welcome messages in different languages
        welcome_texts = {
            Language.ENGLISH: (
                "👋 *Welcome to YouTube Summarizer Bot!*\n\n"
                "I can help you understand YouTube videos faster:\n"
                "• 📹 Send me any YouTube link\n"
                "• 📝 Get structured summary with key points\n"
                "• 💬 Ask questions about the video\n"
                "• 🌐 Get responses in your language\n\n"
                "Try it now! Send a YouTube link."
            ),
            Language.HINDI: (
                "👋 *यूट्यूब समरी बॉट में आपका स्वागत है!*\n\n"
                "मैं यूट्यूब वीडियो को तेजी से समझने में मदद कर सकता हूं:\n"
                "• 📹 मुझे कोई यूट्यूब लिंक भेजें\n"
                "• 📝 मुख्य बिंदुओं के साथ सारांश प्राप्त करें\n"
                "• 💬 वीडियो के बारे में सवाल पूछें\n"
                "• 🌐 अपनी भाषा में जवाब पाएं\n\n"
                "अभी试试 करें! कोई यूट्यूब लिंक भेजें।"
            ),
            Language.TAMIL: (
                "👋 *யூடியூப் சுருக்க போட்டுக்கு வரவேற்கிறோம்!*\n\n"
                "நான் YouTube வீடியோக்களை விரைவாக புரிந்துகொள்ள உதவுகிறேன்:\n"
                "• 📹 எனக்கு YouTube இணைப்பை அனுப்புங்கள்\n"
                "• 📝 முக்கிய புள்ளிகளுடன் சுருக்கத்தைப் பெறுங்கள்\n"
                "• 💬 வீடியோ பற்றி கேள்விகள் கேளுங்கள்\n"
                "• 🌐 உங்கள் மொழியில் பதில்களைப் பெறுங்கள்\n\n"
                "இப்போதே试试! YouTube இணைப்பை அனுப்புங்கள்."
            ),
        }
        
        # Default to English if language not found
        welcome = welcome_texts.get(language, welcome_texts[Language.ENGLISH])
        
        # Add language tip
        if language == Language.ENGLISH:
            welcome += "\n\n💡 *Tip:* Use /language to switch to Hindi, Tamil, and more!"
        else:
            welcome += "\n\n💡 *सुझाव:* भाषा बदलने के लिए /language का उपयोग करें!"
        
        return welcome
    
    async def _handle_help(
        self,
        context: UpdateContext,
        session: Optional[UserSession],
        args: str,
    ) -> Optional[str]:
        """
        Handle /help command.
        
        Shows available commands and usage instructions.
        
        Args:
            context: Update context
            session: User session
            args: Command arguments
            
        Returns:
            Help text
        """
        # If specific command help requested
        if args:
            command = args.strip().lower()
            cmd_info = self._get_command(command)
            
            if cmd_info:
                return self._get_command_help(command, cmd_info)
            else:
                return f"Command '/{command}' not found. Use /help to see all commands."
        
        # Return general help
        return self._help_text
    
    def _get_command_help(self, command: str, cmd_info: Dict[str, Any]) -> str:
        """
        Get detailed help for a specific command.
        
        Args:
            command: Command name
            cmd_info: Command information
            
        Returns:
            Detailed help text
        """
        lines = [f"📖 *Help: /{command}*\n"]
        lines.append(f"*Description:* {cmd_info['description']}")
        
        if cmd_info.get('aliases'):
            aliases = ", ".join(f"/{a}" for a in cmd_info['aliases'])
            lines.append(f"*Aliases:* {aliases}")
        
        # Command-specific details
        if command == "language":
            lines.extend([
                "",
                "*Usage:* `/language [language_name]`",
                "*Examples:*",
                "• `/language hindi` - Switch to Hindi",
                "• `/language ta` - Switch to Tamil (using language code)",
                "• `/language` - Show language selection menu",
                "",
                "*Supported languages:*",
                "English (en), हिन्दी (hi), தமிழ் (ta), తెలుగు (te), ಕನ್ನಡ (kn)",
            ])
        elif command == "start":
            lines.extend([
                "",
                "*Usage:* `/start`",
                "Initialize the bot and create your session.",
            ])
        elif command == "help":
            lines.extend([
                "",
                "*Usage:* `/help [command]`",
                "Show help for all commands or a specific command.",
            ])
        elif command == "summary":
            lines.extend([
                "",
                "*Usage:* `/summary`",
                "Show the last video summary again (after you have sent a YouTube link).",
            ])
        elif command == "deepdive":
            lines.extend([
                "",
                "*Usage:* `/deepdive`",
                "Generate a detailed summary for the current video. Send a link first.",
            ])
        elif command == "actionpoints":
            lines.extend([
                "",
                "*Usage:* `/actionpoints`",
                "Show action items extracted from the last video. Send a link first.",
            ])

        return "\n".join(lines)
    
    async def _handle_language(
        self,
        context: UpdateContext,
        session: UserSession,
        args: str,
    ) -> Optional[str]:
        """
        Handle /language command.
        
        Allows user to change their preferred language.
        
        Args:
            context: Update context
            session: User session
            args: Language name or code
            
        Returns:
            Response text or None if keyboard sent
        """
        # If no args, show language selection keyboard
        if not args:
            return await self._show_language_keyboard(context, session)
        
        # Parse language from args
        target_lang = args.strip().lower()
        language = self._parse_language_input(target_lang)
        
        if not language:
            return (
                f"❌ Language '{target_lang}' not supported.\n\n"
                f"Supported languages: English, हिन्दी, தமிழ், తెలుగు, ಕನ್ನಡ\n"
                f"Use language codes: en, hi, ta, te, kn"
            )
        
        # Update session language
        old_language = session.language
        session.language = language
        session.modified = True
        
        # Confirm change in new language
        confirm_texts = {
            Language.ENGLISH: f"✅ Language changed to English",
            Language.HINDI: f"✅ भाषा बदलकर हिन्दी कर दी गई",
            Language.TAMIL: f"✅ மொழி தமிழுக்கு மாற்றப்பட்டது",
            Language.TELUGU: f"✅ భాష తెలుగుకు మార్చబడింది",
            Language.KANNADA: f"✅ ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ",
        }
        
        # Get confirmation in new language
        confirm = confirm_texts.get(language, confirm_texts[Language.ENGLISH])
        
        # Add note about old language if changed
        if old_language != language:
            confirm += f"\n\n💡 Previously: {old_language.value}"
        
        return confirm

    async def _handle_summary(
        self,
        context: UpdateContext,
        session: Optional[UserSession],
        args: str,
    ) -> Optional[str]:
        """Show the last video summary again (from session)."""
        if not session or not session.current_video_id:
            return "📎 Send a YouTube link first to get a summary. Then use /summary to see it again."
        last = (session.metadata or {}).get("last_summary")
        if not last or not last.get("key_points") and not last.get("core_takeaway"):
            return "📎 Send a YouTube link first to get a summary. Then use /summary to see it again."
        return self._format_last_summary(last)

    def _format_last_summary(self, last: Dict[str, Any]) -> str:
        """Format last_summary dict as message text."""
        lines = [f"📹 *{last.get('title') or 'Video summary'}*\n", "🔑 *Key Points:*"]
        for i, pt in enumerate(last.get("key_points") or [], 1):
            p = (pt.get("point") or "").strip() or "(No description)"
            ts = pt.get("timestamp") or ""
            lines.append(f"{i}. `{ts}` - {p}" if ts else f"{i}. {p}")
        lines.append("\n💡 *Core insight:*")
        lines.append((last.get("core_takeaway") or "See key points above.").strip())
        return "\n".join(lines)

    async def _handle_deepdive(
        self,
        context: UpdateContext,
        session: Optional[UserSession],
        args: str,
    ) -> Optional[str]:
        """Generate a detailed summary for the current video."""
        if not session or not session.current_video_id:
            return "📎 Send a YouTube link first. Then use /deepdive for a detailed summary."
        if not self.summarizer_service or not self.youtube_service:
            return "⚠️ Detailed summary is not available (service not configured)."
        video_id = session.current_video_id
        try:
            transcript = await self.youtube_service.get_transcript(video_id)
            metadata = await self.youtube_service.get_video_metadata(video_id)
            summary = await self.summarizer_service.generate_summary(
                transcript=transcript,
                metadata=metadata,
                summary_type=SummaryType.DETAILED,
                language=session.language,
            )
            if session.language != Language.ENGLISH:
                summary = await self.language_service.translate_summary(
                    summary=summary,
                    target_language=session.language,
                )
            lines = [f"📹 *{metadata.get('title') or 'Video'}* — _Detailed summary_\n", "🔑 *Key Points:*"]
            for i, pt in enumerate(summary.key_points or [], 1):
                p = (pt.get("point") or "").strip() or "(No description)"
                ts = pt.get("timestamp") or ""
                lines.append(f"{i}. `{ts}` - {p}" if ts else f"{i}. {p}")
            lines.append("\n💡 *Core insight:*")
            lines.append((summary.core_takeaway or "See key points above.").strip())
            action_items = getattr(summary, "action_items", None) or []
            if action_items:
                lines.append("\n✅ *Action Items:*")
                for j, item in enumerate(action_items, 1):
                    lines.append(f"  {j}. {item}")
            return "\n".join(lines)
        except Exception as e:
            self.logger.warning("command_handler.deepdive_failed", video_id=video_id, error=str(e))
            return f"⚠️ Could not generate detailed summary: {str(e)[:200]}. Try again or send the link again."

    async def _handle_actionpoints(
        self,
        context: UpdateContext,
        session: Optional[UserSession],
        args: str,
    ) -> Optional[str]:
        """Show action items from the last video."""
        if not session or not session.current_video_id:
            return "📎 Send a YouTube link first. Then use /actionpoints to see action items."
        last = (session.metadata or {}).get("last_summary")
        if not last:
            return "📎 Send a YouTube link first to get a summary. Then use /actionpoints."
        action_items = last.get("action_items") or []
        if not action_items:
            return "📋 No action items were extracted for the last video.\n\n💡 Try /deepdive for a more detailed summary."
        title = last.get("title") or "Last video"
        lines = [f"✅ *Action Items* — {title}\n"]
        for i, item in enumerate(action_items, 1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)

    async def _show_language_keyboard(
        self,
        context: UpdateContext,
        session: UserSession,
    ) -> None:
        """
        Show interactive language selection keyboard.
        
        Args:
            context: Update context
            session: User session
        """
        # Language options with flags
        languages = [
            {"code": "en", "name": "🇬🇧 English", "callback": "lang_en"},
            {"code": "hi", "name": "🇮🇳 हिन्दी", "callback": "lang_hi"},
            {"code": "ta", "name": "🇮🇳 தமிழ்", "callback": "lang_ta"},
            {"code": "te", "name": "🇮🇳 తెలుగు", "callback": "lang_te"},
            {"code": "kn", "name": "🇮🇳 ಕನ್ನಡ", "callback": "lang_kn"},
        ]
        
        # Build keyboard (2 columns)
        keyboard = []
        row = []
        for i, lang in enumerate(languages):
            # Highlight current language
            if lang["code"] == session.language.code:
                lang["name"] = f"✅ {lang['name']}"
            
            row.append({
                "text": lang["name"],
                "callback_data": lang["callback"],
            })
            
            if len(row) == 2 or i == len(languages) - 1:
                keyboard.append(row)
                row = []
        
        # Add close button
        keyboard.append([{
            "text": "❌ Close",
            "callback_data": "close_language_menu",
        }])
        
        # Create inline keyboard
        reply_markup = KeyboardBuilder.build(keyboard, "inline")
        
        # Send message with keyboard
        await self.telegram_client.send_message(
            chat_id=context.chat_id,
            text="🌐 *Select your preferred language:*\n\nChoose the language for summaries and responses.",
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    
    def _parse_language_input(self, input_str: str) -> Optional[Language]:
        """
        Parse user input to Language enum.
        
        Args:
            input_str: Language name or code
            
        Returns:
            Language enum or None if not found
        """
        # Map of names and codes to Language
        language_map = {
            # English variations
            "en": Language.ENGLISH,
            "english": Language.ENGLISH,
            "eng": Language.ENGLISH,
            "inglish": Language.ENGLISH,
            
            # Hindi variations
            "hi": Language.HINDI,
            "hindi": Language.HINDI,
            "हिन्दी": Language.HINDI,
            "हिंदी": Language.HINDI,
            
            # Tamil variations
            "ta": Language.TAMIL,
            "tamil": Language.TAMIL,
            "தமிழ்": Language.TAMIL,
            
            # Telugu variations
            "te": Language.TELUGU,
            "telugu": Language.TELUGU,
            "తెలుగు": Language.TELUGU,
            
            # Kannada variations
            "kn": Language.KANNADA,
            "kannada": Language.KANNADA,
            "ಕನ್ನಡ": Language.KANNADA,
        }
        
        return language_map.get(input_str.lower())
    
    async def _handle_unknown_command(
        self,
        context: UpdateContext,
        command: str,
    ) -> None:
        """
        Handle unknown commands.
        
        Args:
            context: Update context
            command: Unknown command name
        """
        # Get user's language from session if available
        language = Language.ENGLISH
        session = await self.session_store.get_session(context.user_id)
        if session:
            language = session.language
        
        # Response in user's language
        if language == Language.ENGLISH:
            response = (
                f"❌ Unknown command: /{command}\n\n"
                f"Use /help to see available commands."
            )
        elif language == Language.HINDI:
            response = (
                f"❌ अज्ञात कमांड: /{command}\n\n"
                f"उपलब्ध कमांड देखने के लिए /help का उपयोग करें।"
            )
        elif language == Language.TAMIL:
            response = (
                f"❌ அறியப்படாத கட்டளை: /{command}\n\n"
                f"கிடைக்கக்கூடிய கட்டளைகளைப் பார்க்க /help பயன்படுத்தவும்."
            )
        else:
            response = f"❌ Unknown command: /{command}\n\nUse /help to see available commands."
        
        await self._send_response(
            chat_id=context.chat_id,
            text=response,
            reply_to_message_id=context.update.effective_message.message_id,
        )
    
    # ------------------------------------------------------------------------
    # Callback Query Handlers
    # ------------------------------------------------------------------------
    
    async def handle_callback(self, context: UpdateContext) -> None:
        """
        Handle callback queries from inline keyboards.
        
        Args:
            context: Update context with callback query
        """
        callback_query = context.update.callback_query
        data = callback_query.data
        
        if data.startswith("lang_"):
            # Language selection callback
            lang_code = data.split("_")[1]
            await self._handle_language_callback(context, lang_code)
        
        elif data == "close_language_menu":
            await callback_query.answer()
            await callback_query.delete_message()
    
    async def _handle_language_callback(
        self,
        context: UpdateContext,
        lang_code: str,
    ) -> None:
        """
        Handle language selection from keyboard.
        
        Args:
            context: Update context
            lang_code: Selected language code
        """
        callback_query = context.update.callback_query
        user_id = context.user_id
        
        # Map code to Language
        code_to_lang = {
            "en": Language.ENGLISH,
            "hi": Language.HINDI,
            "ta": Language.TAMIL,
            "te": Language.TELUGU,
            "kn": Language.KANNADA,
        }
        
        language = code_to_lang.get(lang_code)
        if not language:
            await callback_query.answer("Invalid language", show_alert=True)
            return
        
        # Get session
        session = await self.session_store.get_session(user_id)
        if not session:
            session = await self.session_store.create_session(
                user_id=user_id,
                chat_id=context.chat_id,
            )
        
        # Update language
        old_language = session.language
        session.language = language
        session.modified = True
        await self.session_store.save_session(session)
        
        # Answer callback
        await callback_query.answer(f"Language changed to {language.value}")
        
        # Update message to show selection
        confirm_texts = {
            Language.ENGLISH: f"✅ Language changed to English",
            Language.HINDI: f"✅ भाषा बदलकर हिन्दी कर दी गई",
            Language.TAMIL: f"✅ மொழி தமிழுக்கு மாற்றப்பட்டது",
            Language.TELUGU: f"✅ భాష తెలుగుకు మార్చబడింది",
            Language.KANNADA: f"✅ ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ",
        }
        
        confirm = confirm_texts.get(language, confirm_texts[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=f"🌐 {confirm}",
            parse_mode="Markdown",
        )


# ------------------------------------------------------------------------
# Example Session Store Interface (for reference)
# ------------------------------------------------------------------------

class UserSession:
    """User session data structure"""
    
    def __init__(
        self,
        user_id: int,
        chat_id: int,
        language: Language = Language.ENGLISH,
        current_video_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.language = language
        self.current_video_id = current_video_id
        self.created_at = created_at or datetime.now()
        self.updated_at = datetime.now()
        self.modified = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "language": self.language.code,
            "current_video_id": self.current_video_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Create from dictionary"""
        session = cls(
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            language=Language.from_code(data.get("language", "en")),
            current_video_id=data.get("current_video_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
        )
        session.updated_at = datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        return session