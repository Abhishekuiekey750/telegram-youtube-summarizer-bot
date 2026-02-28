"""
Domain value objects
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class Language(Enum):
    """Supported languages for summaries and Q&A"""
    ENGLISH = ("en", "English", "English")
    HINDI = ("hi", "Hindi", "हिन्दी")
    TAMIL = ("ta", "Tamil", "தமிழ்")
    TELUGU = ("te", "Telugu", "తెలుగు")
    KANNADA = ("kn", "Kannada", "ಕನ್ನಡ")
    MALAYALAM = ("ml", "Malayalam", "മലയാളം")
    BENGALI = ("bn", "Bengali", "বাংলা")
    MARATHI = ("mr", "Marathi", "मराठी")
    GUJARATI = ("gu", "Gujarati", "ગુજરાતી")
    PUNJABI = ("pa", "Punjabi", "ਪੰਜਾਬੀ")
    URDU = ("ur", "Urdu", "اردو")
    SANSKRIT = ("sa", "Sanskrit", "संस्कृत")

    def __init__(self, code: str, name: str, native_name: str):
        self._code = code
        self._name = name
        self._native_name = native_name

    @property
    def code(self) -> str:
        return self._code

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_name(self) -> str:
        return self._native_name

    @property
    def value(self) -> str:
        return self._native_name

    @classmethod
    def from_code(cls, code: str) -> "Language":
        """Get Language from ISO code"""
        code_lower = (code or "en").lower().strip()
        for lang in cls:
            if lang.code == code_lower:
                return lang
        return cls.ENGLISH


@dataclass(frozen=True)
class VideoId:
    """YouTube video ID value object"""
    value: str

    def __str__(self) -> str:
        return self.value
