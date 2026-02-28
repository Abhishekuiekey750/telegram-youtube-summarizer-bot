"""
Inline Callback Handler
Processes all callback queries from inline keyboards

Supported callbacks:
- Language selection (lang_*)
- Summary options (summary_*)
- Action buttons (action_*)
- Menu navigation (menu_*)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import structlog

from internal.bot.handlers.base import BaseHandler
from internal.bot.dispatcher import UpdateContext
from internal.bot.keyboard import KeyboardBuilder
from internal.services.language import LanguageService
from internal.services.summarizer import SummarizerService, SummaryType
from internal.services.youtube import YouTubeService
from internal.storage.session import SessionStore, UserSession
from internal.domain.value_objects import Language
from internal.pkg.errors import ValidationError, ErrorKind, NotFoundError


class CallbackHandler(BaseHandler):
    """
    Handles all inline keyboard callback queries.
    
    Supports:
    - Language selection with instant feedback
    - Summary type toggling (short, detailed, bullets)
    - Action buttons (ask question, new video, etc.)
    - Menu navigation
    
    Features:
    - Session management for user preferences
    - Instant callback acknowledgment
    - Message editing for seamless UX
    - Multilingual responses
    - Metrics tracking
    """
    
    # Callback pattern definitions
    CALLBACK_PATTERNS = {
        # Language: lang_en, lang_hi, lang_ta, etc.
        "language": re.compile(r"^lang_([a-z]{2})$"),
        
        # Summary: summary_short_video123, summary_detailed_video123
        "summary": re.compile(r"^summary_([a-z]+)_(.+)$"),
        
        # Actions: action_ask, action_new, action_deepdive
        "action": re.compile(r"^action_([a-z_]+)$"),
        
        # Menu: menu_back, menu_main, menu_language
        "menu": re.compile(r"^menu_([a-z_]+)$"),
    }
    
    def __init__(
        self,
        telegram_client,
        session_store: SessionStore,
        language_service: LanguageService,
        summarizer_service: SummarizerService,
        youtube_service: YouTubeService,
        logger=None,
        metrics=None,
    ):
        """
        Initialize callback handler.
        
        Args:
            telegram_client: Client for sending messages
            session_store: Session manager
            language_service: Language service
            summarizer_service: Summarizer service
            youtube_service: YouTube service
            logger: Structured logger
            metrics: Metrics collector
        """
        super().__init__(telegram_client, logger, metrics)
        
        self.session_store = session_store
        self.language_service = language_service
        self.summarizer_service = summarizer_service
        self.youtube_service = youtube_service
        
        # Response templates
        self.action_responses = self._init_action_responses()
        
        self.logger.info(
            "callback_handler.initialized",
            patterns=list(self.CALLBACK_PATTERNS.keys()),
        )
    
    @property
    def handler_name(self) -> str:
        return "callback"
    
    async def handle(self, context: UpdateContext) -> None:
        """
        Main entry point for callback queries.
        
        Args:
            context: Update context with callback_query
        """
        start_time = datetime.now()
        
        try:
            await self.before_handle(context)
            
            # Validate callback query
            if not context.update.callback_query:
                return
            
            callback_query = context.update.callback_query
            callback_data = callback_query.data
            
            self.logger.debug(
                "callback_handler.received",
                user_id=context.user_id,
                callback_data=callback_data,
            )
            
            # Always answer callback immediately (removes loading state)
            await callback_query.answer()
            
            # Load user session
            session = await self.session_store.get_session(context.user_id)
            if not session:
                session = await self.session_store.create_session(
                    user_id=context.user_id,
                    chat_id=context.chat_id,
                )
            
            # Add language to context
            context.metadata["language"] = session.language
            
            # Route to appropriate handler
            handled = await self._route_callback(
                callback_data=callback_data,
                callback_query=callback_query,
                session=session,
                context=context,
            )
            
            if not handled:
                self.logger.warning(
                    "callback_handler.unhandled",
                    callback_data=callback_data,
                    user_id=context.user_id,
                )
                await self._send_unhandled_response(callback_query, session.language)
            
            # Track metrics
            self.metrics.increment(
                "callbacks.processed",
                tags={"type": callback_data.split("_")[0]},
            )
            
            await self.after_handle(context)
            
        except Exception as e:
            self.logger.exception(
                "callback_handler.error",
                user_id=context.user_id,
                error=str(e),
            )
            await self._send_error_response(context.update.callback_query)
            raise
        finally:
            self._track_processing_time(start_time)
    
    # ------------------------------------------------------------------------
    # Callback Routing
    # ------------------------------------------------------------------------
    
    async def _route_callback(
        self,
        callback_data: str,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> bool:
        """
        Route callback to appropriate handler based on pattern.
        
        Args:
            callback_data: Raw callback data
            callback_query: Telegram callback query object
            session: User session
            context: Update context
            
        Returns:
            True if handled, False otherwise
        """
        # Try each pattern
        for handler_type, pattern in self.CALLBACK_PATTERNS.items():
            match = pattern.match(callback_data)
            if not match:
                continue
            
            if handler_type == "language":
                lang_code = match.group(1)
                await self._handle_language_callback(
                    callback_query=callback_query,
                    session=session,
                    lang_code=lang_code,
                )
                return True
                
            elif handler_type == "summary":
                summary_type = match.group(1)
                video_id = match.group(2)
                await self._handle_summary_callback(
                    callback_query=callback_query,
                    session=session,
                    summary_type=summary_type,
                    video_id=video_id,
                )
                return True
                
            elif handler_type == "action":
                action = match.group(1)
                await self._handle_action_callback(
                    callback_query=callback_query,
                    session=session,
                    action=action,
                    context=context,
                )
                return True
                
            elif handler_type == "menu":
                menu_item = match.group(1)
                await self._handle_menu_callback(
                    callback_query=callback_query,
                    session=session,
                    menu_item=menu_item,
                )
                return True
        
        return False
    
    # ------------------------------------------------------------------------
    # Language Callback Handlers
    # ------------------------------------------------------------------------
    
    async def _handle_language_callback(
        self,
        callback_query,
        session: UserSession,
        lang_code: str,
    ) -> None:
        """
        Handle language selection callback.
        
        Args:
            callback_query: Telegram callback query
            session: User session
            lang_code: Selected language code (en, hi, ta, etc.)
        """
        # Map code to Language
        language = Language.from_code(lang_code)
        if not language:
            await callback_query.answer(
                text="Invalid language",
                show_alert=True,
            )
            return
        
        # Store old language for message
        old_language = session.language
        
        # Update session
        session.language = language
        session.modified = True
        await self.session_store.save_session(session)
        
        # Get confirmation message in new language
        confirm_text = self._get_language_confirmation(language, old_language)
        
        # Update the message that had the language keyboard
        await callback_query.edit_message_text(
            text=confirm_text,
            parse_mode="Markdown",
            reply_markup=None,  # Remove keyboard
        )
        
        # Send follow-up prompt in new language
        followup_text = self._get_language_followup(language)
        await self.telegram_client.send_message(
            chat_id=callback_query.message.chat_id,
            text=followup_text,
        )
        
        self.logger.info(
            "callback_handler.language_changed",
            user_id=session.user_id,
            old_language=old_language.code,
            new_language=language.code,
        )
    
    def _get_language_confirmation(self, language: Language, old_language: Language) -> str:
        """Get language confirmation message"""
        
        confirmations = {
            Language.ENGLISH: f"✅ *Language changed to English*\n\nPreviously: {old_language.value}",
            Language.HINDI: f"✅ *भाषा बदलकर हिन्दी कर दी गई*\n\nपहले: {old_language.value}",
            Language.TAMIL: f"✅ *மொழி தமிழுக்கு மாற்றப்பட்டது*\n\nமுன்பு: {old_language.value}",
            Language.TELUGU: f"✅ *భాష తెలుగుకు మార్చబడింది*\n\nగతంలో: {old_language.value}",
            Language.KANNADA: f"✅ *ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ*\n\nಹಿಂದೆ: {old_language.value}",
            Language.MALAYALAM: f"✅ *ഭാഷ മലയാളത്തിലേക്ക് മാറ്റി*\n\nമുമ്പ്: {old_language.value}",
            Language.BENGALI: f"✅ *ভাষা বাংলায় পরিবর্তন করা হয়েছে*\n\nপূর্বে: {old_language.value}",
        }
        
        return confirmations.get(language, confirmations[Language.ENGLISH])
    
    def _get_language_followup(self, language: Language) -> str:
        """Get follow-up message after language change"""
        
        followups = {
            Language.ENGLISH: "You can now ask questions about videos in English!",
            Language.HINDI: "अब आप हिन्दी में वीडियो के बारे में सवाल पूछ सकते हैं!",
            Language.TAMIL: "இப்போது நீங்கள் தமிழில் வீடியோக்கள் பற்றி கேள்விகள் கேட்கலாம்!",
            Language.TELUGU: "ఇప్పుడు మీరు తెలుగులో వీడియోల గురించి ప్రశ్నలు అడగవచ్చు!",
            Language.KANNADA: "ಈಗ ನೀವು ಕನ್ನಡದಲ್ಲಿ ವೀಡಿಯೊಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಬಹುದು!",
        }
        
        return followups.get(language, followups[Language.ENGLISH])
    
    # ------------------------------------------------------------------------
    # Summary Option Callback Handlers
    # ------------------------------------------------------------------------
    
    async def _handle_summary_callback(
        self,
        callback_query,
        session: UserSession,
        summary_type: str,
        video_id: str,
    ) -> None:
        """
        Handle summary option selection.
        
        Args:
            callback_query: Telegram callback query
            session: User session
            summary_type: Type of summary requested
            video_id: YouTube video ID
        """
        # Map summary type string to enum
        type_map = {
            "short": SummaryType.CONCISE,
            "detailed": SummaryType.DETAILED,
            "bullets": SummaryType.BULLET_POINTS,
            "timestamps": SummaryType.TIMESTAMPS,
        }
        
        summary_type_enum = type_map.get(summary_type)
        if not summary_type_enum:
            await callback_query.answer(
                text="Invalid summary type",
                show_alert=True,
            )
            return
        
        # Show loading state
        loading_text = self._get_loading_text(session.language)
        await callback_query.edit_message_text(
            text=loading_text,
            reply_markup=None,
        )
        
        try:
            # Fetch video metadata
            metadata = await self.youtube_service.get_video_metadata(video_id)
            
            # Generate summary of requested type
            summary = await self.summarizer_service.generate_summary(
                video_id=video_id,
                summary_type=summary_type_enum,
                language=session.language,
            )
            
            # Format response based on summary type
            response_text = self._format_summary_response(
                summary=summary,
                metadata=metadata,
                summary_type=summary_type_enum,
                language=session.language,
            )
            
            # Send summary
            await self.telegram_client.send_message(
                chat_id=callback_query.message.chat_id,
                text=response_text,
                parse_mode="Markdown",
            )
            
            # Show follow-up keyboard
            await self._show_summary_followup(
                chat_id=callback_query.message.chat_id,
                video_id=video_id,
                language=session.language,
            )
            
            self.metrics.increment(
                "summaries.generated",
                tags={"type": summary_type},
            )
            
        except Exception as e:
            self.logger.exception(
                "callback_handler.summary_failed",
                video_id=video_id,
                error=str(e),
            )
            
            error_text = self._get_error_text(session.language)
            await self.telegram_client.send_message(
                chat_id=callback_query.message.chat_id,
                text=error_text,
            )
    
    def _format_summary_response(
        self,
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
        summary_type: SummaryType,
        language: Language,
    ) -> str:
        """Format summary based on type"""
        
        lines = []
        
        # Header
        if language == Language.ENGLISH:
            lines.append(f"📹 *{metadata['title']}*\n")
        elif language == Language.HINDI:
            lines.append(f"📹 *{metadata['title']}*\n")
        
        if summary_type == SummaryType.CONCISE:
            lines.append(summary.get("concise_summary", ""))
            
        elif summary_type == SummaryType.DETAILED:
            lines.append(summary.get("detailed_summary", ""))
            
        elif summary_type == SummaryType.BULLET_POINTS:
            if language == Language.ENGLISH:
                lines.append("🔑 *Key Points:*\n")
            elif language == Language.HINDI:
                lines.append("🔑 *मुख्य बिंदु:*\n")
            
            for i, point in enumerate(summary.get("key_points", []), 1):
                lines.append(f"{i}. {point}")
                
        elif summary_type == SummaryType.TIMESTAMPS:
            if language == Language.ENGLISH:
                lines.append("⏱️ *Important Moments:*\n")
            elif language == Language.HINDI:
                lines.append("⏱️ *महत्वपूर्ण क्षण:*\n")
            
            for moment in summary.get("timestamps", []):
                lines.append(f"`{moment['timestamp']}` - {moment['description']}")
        
        return "\n".join(lines)
    
    # ------------------------------------------------------------------------
    # Action Callback Handlers
    # ------------------------------------------------------------------------
    
    async def _handle_action_callback(
        self,
        callback_query,
        session: UserSession,
        action: str,
        context: UpdateContext,
    ) -> None:
        """
        Handle action button callbacks.
        
        Args:
            callback_query: Telegram callback query
            session: User session
            action: Action to perform
            context: Update context
        """
        # Map actions to handlers
        action_handlers = {
            "ask": self._action_ask_question,
            "new": self._action_new_video,
            "deepdive": self._action_deep_dive,
            "language": self._action_show_languages,
            "share": self._action_share,
            "save": self._action_save,
            "feedback": self._action_feedback,
        }
        
        handler = action_handlers.get(action)
        if handler:
            await handler(callback_query, session, context)
        else:
            await self._action_unknown(callback_query, session, action)
    
    async def _action_ask_question(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle ask question action"""
        
        # Edit the message to prompt for question
        prompts = {
            Language.ENGLISH: "💬 *Ask a question*\n\nSend me your question about this video:",
            Language.HINDI: "💬 *सवाल पूछें*\n\nइस वीडियो के बारे में अपना सवाल भेजें:",
            Language.TAMIL: "💬 *கேள்வி கேளுங்கள்*\n\nஇந்த வீடியோ பற்றிய உங்கள் கேள்வியை அனுப்புங்கள்:",
        }
        
        prompt = prompts.get(session.language, prompts[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=prompt,
            parse_mode="Markdown",
        )
        
        # Set session state to expecting question
        if not session.metadata:
            session.metadata = {}
        session.metadata["expecting_question"] = True
        session.modified = True
        await self.session_store.save_session(session)
    
    async def _action_new_video(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle new video action"""
        
        messages = {
            Language.ENGLISH: "📹 *New Video*\n\nSend me another YouTube link to summarize!",
            Language.HINDI: "📹 *नया वीडियो*\n\nमुझे सारांश के लिए एक और YouTube लिंक भेजें!",
            Language.TAMIL: "📹 *புதிய வீடியோ*\n\nசுருக்கமாக மற்றொரு YouTube இணைப்பை அனுப்புங்கள்!",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
    
    async def _action_deep_dive(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle deep dive action"""
        
        if not session.current_video_id:
            await self._action_new_video(callback_query, session, context)
            return
        
        messages = {
            Language.ENGLISH: "🔍 *Deep Dive Mode*\n\nAsk detailed questions and I'll search deeply through the video transcript.",
            Language.HINDI: "🔍 *गहराई से जानकारी मोड*\n\nविस्तृत सवाल पूछें और मैं वीडियो ट्रांसक्रिप्ट में गहराई से खोज करूंगा।",
            Language.TAMIL: "🔍 *ஆழமான பயன்முறை*\n\nவிரிவான கேள்விகளைக் கேளுங்கள், நான் வீடியோ டிரான்ஸ்கிரிப்டில் ஆழமாகத் தேடுகிறேன்.",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
        
        # Set deep dive mode in session
        if not session.metadata:
            session.metadata = {}
        session.metadata["mode"] = "deep_dive"
        session.modified = True
        await self.session_store.save_session(session)
    
    async def _action_show_languages(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Show language selection keyboard"""
        
        # Build language keyboard
        keyboard = self._build_language_keyboard()
        
        messages = {
            Language.ENGLISH: "🌐 *Select Language*\n\nChoose your preferred language:",
            Language.HINDI: "🌐 *भाषा चुनें*\n\nअपनी पसंदीदा भाषा चुनें:",
            Language.TAMIL: "🌐 *மொழியைத் தேர்ந்தெடுக்கவும்*\n\nஉங்கள் விருப்பமான மொழியைத் தேர்வுசெய்க:",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
    
    async def _action_share(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle share action"""
        
        messages = {
            Language.ENGLISH: "📤 *Share*\n\nShare this bot with friends: https://t.me/your_bot",
            Language.HINDI: "📤 *शेयर करें*\n\nइस बॉट को दोस्तों के साथ साझा करें: https://t.me/your_bot",
            Language.TAMIL: "📤 *பகிரவும்*\n\nஇந்த போட்டை நண்பர்களுடன் பகிரவும்: https://t.me/your_bot",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
    
    async def _action_save(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle save action"""
        
        messages = {
            Language.ENGLISH: "💾 *Saved*\n\nThis summary has been saved to your history.",
            Language.HINDI: "💾 *सेव हो गया*\n\nयह सारांश आपके इतिहास में सहेज लिया गया है।",
            Language.TAMIL: "💾 *சேமிக்கப்பட்டது*\n\nஇந்த சுருக்கம் உங்கள் வரலாற்றில் சேமிக்கப்பட்டுள்ளது.",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
    
    async def _action_feedback(
        self,
        callback_query,
        session: UserSession,
        context: UpdateContext,
    ) -> None:
        """Handle feedback action"""
        
        messages = {
            Language.ENGLISH: "📝 *Feedback*\n\nPlease send your feedback or suggestions:",
            Language.HINDI: "📝 *फीडबैक*\n\nकृपया अपना फीडबैक या सुझाव भेजें:",
            Language.TAMIL: "📝 *கருத்து*\n\nஉங்கள் கருத்து அல்லது பரிந்துரைகளை அனுப்பவும்:",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
        
        # Set session state to expecting feedback
        if not session.metadata:
            session.metadata = {}
        session.metadata["expecting_feedback"] = True
        session.modified = True
        await self.session_store.save_session(session)
    
    async def _action_unknown(
        self,
        callback_query,
        session: UserSession,
        action: str,
    ) -> None:
        """Handle unknown action"""
        
        messages = {
            Language.ENGLISH: f"❌ Unknown action: {action}",
            Language.HINDI: f"❌ अज्ञात कार्रवाई: {action}",
            Language.TAMIL: f"❌ அறியப்படாத செயல்: {action}",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
        )
    
    # ------------------------------------------------------------------------
    # Menu Callback Handlers
    # ------------------------------------------------------------------------
    
    async def _handle_menu_callback(
        self,
        callback_query,
        session: UserSession,
        menu_item: str,
    ) -> None:
        """
        Handle menu navigation callbacks.
        
        Args:
            callback_query: Telegram callback query
            session: User session
            menu_item: Menu item to navigate to
        """
        if menu_item == "main":
            await self._show_main_menu(callback_query, session)
        elif menu_item == "back":
            await self._go_back(callback_query, session)
        elif menu_item == "language":
            await self._show_language_menu(callback_query, session)
        elif menu_item == "help":
            await self._show_help_menu(callback_query, session)
    
    async def _show_main_menu(
        self,
        callback_query,
        session: UserSession,
    ) -> None:
        """Show main menu"""
        
        keyboard = [
            [
                {"text": "📹 New Video", "callback_data": "action_new"},
                {"text": "❓ Ask", "callback_data": "action_ask"},
            ],
            [
                {"text": "🔍 Deep Dive", "callback_data": "action_deepdive"},
                {"text": "🌐 Language", "callback_data": "menu_language"},
            ],
            [
                {"text": "📤 Share", "callback_data": "action_share"},
                {"text": "📝 Feedback", "callback_data": "action_feedback"},
            ],
        ]
        
        reply_markup = KeyboardBuilder.build(keyboard, "inline")
        
        messages = {
            Language.ENGLISH: "📋 *Main Menu*\n\nWhat would you like to do?",
            Language.HINDI: "📋 *मुख्य मेनू*\n\nआप क्या करना चाहेंगे?",
            Language.TAMIL: "📋 *முதன்மை மெனு*\n\nநீங்கள் என்ன செய்ய விரும்புகிறீர்கள்?",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    
    async def _show_language_menu(
        self,
        callback_query,
        session: UserSession,
    ) -> None:
        """Show language selection menu"""
        
        keyboard = self._build_language_keyboard()
        
        # Add back button
        keyboard.append([{
            "text": "🔙 Back to Main",
            "callback_data": "menu_main",
        }])
        
        reply_markup = KeyboardBuilder.build(keyboard, "inline")
        
        messages = {
            Language.ENGLISH: "🌐 *Select Language*\n\nChoose your preferred language:",
            Language.HINDI: "🌐 *भाषा चुनें*\n\nअपनी पसंदीदा भाषा चुनें:",
            Language.TAMIL: "🌐 *மொழியைத் தேர்ந்தெடுக்கவும்*\n\nஉங்கள் விருப்பமான மொழியைத் தேர்வுசெய்க:",
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    
    async def _show_help_menu(
        self,
        callback_query,
        session: UserSession,
    ) -> None:
        """Show help menu"""
        
        messages = {
            Language.ENGLISH: (
                "❓ *Help*\n\n"
                "*How to use:*\n"
                "1. Send a YouTube link\n"
                "2. Get instant summary\n"
                "3. Ask questions about the video\n\n"
                "*Commands:*\n"
                "/start - Start the bot\n"
                "/help - Show this help\n"
                "/language - Change language"
            ),
            Language.HINDI: (
                "❓ *सहायता*\n\n"
                "*उपयोग कैसे करें:*\n"
                "1. YouTube लिंक भेजें\n"
                "2. तुरंत सारांश प्राप्त करें\n"
                "3. वीडियो के बारे में सवाल पूछें\n\n"
                "*कमांड:*\n"
                "/start - बॉट शुरू करें\n"
                "/help - यह सहायता दिखाएं\n"
                "/language - भाषा बदलें"
            ),
        }
        
        message = messages.get(session.language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(
            text=message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Back", callback_data="menu_main")
            ]]),
        )
    
    async def _go_back(
        self,
        callback_query,
        session: UserSession,
    ) -> None:
        """Go back to previous menu"""
        # Implementation would depend on navigation stack
        await self._show_main_menu(callback_query, session)
    
    # ------------------------------------------------------------------------
    # Keyboard Builders
    # ------------------------------------------------------------------------
    
    def _build_language_keyboard(self) -> List[List[Dict[str, str]]]:
        """Build language selection keyboard"""
        
        return [
            [
                {"text": "🇬🇧 English", "callback_data": "lang_en"},
                {"text": "🇮🇳 हिन्दी", "callback_data": "lang_hi"},
            ],
            [
                {"text": "🇮🇳 தமிழ்", "callback_data": "lang_ta"},
                {"text": "🇮🇳 తెలుగు", "callback_data": "lang_te"},
            ],
            [
                {"text": "🇮🇳 ಕನ್ನಡ", "callback_data": "lang_kn"},
                {"text": "🇮🇳 മലയാളം", "callback_data": "lang_ml"},
            ],
            [
                {"text": "🇮🇳 বাংলা", "callback_data": "lang_bn"},
            ],
        ]
    
    async def _show_summary_followup(
        self,
        chat_id: int,
        video_id: str,
        language: Language,
    ) -> None:
        """Show follow-up actions after summary"""
        
        messages = {
            Language.ENGLISH: "📋 *What would you like to do next?*",
            Language.HINDI: "📋 *आगे क्या करना चाहेंगे?*",
            Language.TAMIL: "📋 *அடுத்து என்ன செய்ய விரும்புகிறீர்கள்?*",
        }
        
        message = messages.get(language, messages[Language.ENGLISH])
        
        keyboard = [
            [
                {"text": "❓ Ask Question", "callback_data": "action_ask"},
                {"text": "📹 New Video", "callback_data": "action_new"},
            ],
            [
                {"text": "🔍 Deep Dive", "callback_data": "action_deepdive"},
                {"text": "🌐 Language", "callback_data": "menu_language"},
            ],
        ]
        
        reply_markup = KeyboardBuilder.build(keyboard, "inline")
        
        await self.telegram_client.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    def _init_action_responses(self) -> Dict[str, Dict[Language, str]]:
        """Initialize action response templates"""
        return {
            "ask": {
                Language.ENGLISH: "💬 Send me your question!",
                Language.HINDI: "💬 अपना सवाल भेजें!",
                Language.TAMIL: "💬 உங்கள் கேள்வியை அனுப்புங்கள்!",
            },
            "new": {
                Language.ENGLISH: "📹 Send another YouTube link!",
                Language.HINDI: "📹 एक और YouTube लिंक भेजें!",
                Language.TAMIL: "📹 மற்றொரு YouTube இணைப்பை அனுப்புங்கள்!",
            },
        }
    
    def _get_loading_text(self, language: Language) -> str:
        """Get loading text in appropriate language"""
        
        loading = {
            Language.ENGLISH: "⏳ Processing... Please wait.",
            Language.HINDI: "⏳ प्रोसेस हो रहा है... कृपया प्रतीक्षा करें।",
            Language.TAMIL: "⏳ செயலாக்கப்படுகிறது... தயவுசெய்து காத்திருங்கள்.",
        }
        
        return loading.get(language, loading[Language.ENGLISH])
    
    def _get_error_text(self, language: Language) -> str:
        """Get error text in appropriate language"""
        
        errors = {
            Language.ENGLISH: "❌ Sorry, something went wrong. Please try again.",
            Language.HINDI: "❌ क्षमा करें, कुछ गलत हो गया। कृपया पुनः प्रयास करें।",
            Language.TAMIL: "❌ மன்னிக்கவும், ஏதோ தவறாகிவிட்டது. மீண்டும் முயற்சிக்கவும்.",
        }
        
        return errors.get(language, errors[Language.ENGLISH])
    
    async def _send_unhandled_response(
        self,
        callback_query,
        language: Language,
    ) -> None:
        """Send response for unhandled callback"""
        
        messages = {
            Language.ENGLISH: "This button is no longer valid.",
            Language.HINDI: "यह बटन अब मान्य नहीं है।",
            Language.TAMIL: "இந்த பொத்தான் இப்போது செல்லுபடியாகாது.",
        }
        
        message = messages.get(language, messages[Language.ENGLISH])
        
        await callback_query.edit_message_text(text=message)
    
    async def _send_error_response(self, callback_query) -> None:
        """Send error response"""
        
        await callback_query.edit_message_text(
            text="❌ An error occurred. Please try again.",
        )