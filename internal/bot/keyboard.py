"""
Keyboard builder for Telegram inline and reply keyboards
"""

from typing import List, Dict, Any, Union
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton


class KeyboardBuilder:
    """
    Builds Telegram keyboard markup from button configurations.
    """
    
    @staticmethod
    def build(
        buttons: List[List[Dict[str, str]]],
        keyboard_type: str = "inline",
        **kwargs: Any,
    ) -> Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]:
        """
        Build keyboard markup from button configuration.
        
        Args:
            buttons: List of rows, each row is list of button dicts.
                     Inline: {"text": "...", "callback_data": "..."}
                     Reply: {"text": "..."}
            keyboard_type: "inline" or "reply"
            
        Returns:
            InlineKeyboardMarkup or ReplyKeyboardMarkup
        """
        if keyboard_type == "inline":
            keyboard = []
            for row in buttons:
                kb_row = []
                for btn in row:
                    if isinstance(btn, dict):
                        text = btn.get("text", "")
                        callback_data = btn.get("callback_data", "")
                        url = btn.get("url", "")
                        if url:
                            kb_row.append(InlineKeyboardButton(text=text, url=url))
                        else:
                            kb_row.append(InlineKeyboardButton(text=text, callback_data=callback_data or text))
                    else:
                        kb_row.append(btn)
                keyboard.append(kb_row)
            return InlineKeyboardMarkup(keyboard)
        else:
            # Reply keyboard
            keyboard = []
            for row in buttons:
                kb_row = []
                for btn in row:
                    if isinstance(btn, dict):
                        text = btn.get("text", "")
                        kb_row.append(KeyboardButton(text=text))
                    else:
                        kb_row.append(btn)
                keyboard.append(kb_row)
            return ReplyKeyboardMarkup(keyboard, **kwargs)
