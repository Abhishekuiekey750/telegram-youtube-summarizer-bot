try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.base import BaseConfig


class ConfigLoader:
    def load(self):
        return BaseConfig()