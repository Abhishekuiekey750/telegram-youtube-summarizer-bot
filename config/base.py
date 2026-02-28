import os


class BaseConfig:
    environment = os.environ.get("ENVIRONMENT", "development")

    class telegram:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "DUMMY_TOKEN")
        mode = os.environ.get("TELEGRAM_MODE", "polling")
        allowed_users = []  # Set via TELEGRAM_ALLOWED_USERS (comma-separated)
        webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL")
        webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
        max_connections = 40

    class server:
        host = os.environ.get("SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("SERVER_PORT", "8080"))

    class youtube:
        api_key = os.environ.get("YOUTUBE_API_KEY", "DUMMY_KEY")

    class models:
        class embedding:
            name = "dummy"

        class generation:
            name = "dummy"

    class vector_db:
        pass

    class session:
        pass

    class cache:
        pass

    class language:
        pass