import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

PRICE_MILLS = 499
TOPUPS = (5, 10, 20)
COOKIE = "milieu_session"


@dataclass(frozen=True)
class Settings:
    database_url: str
    public_url: str
    session_secret: str
    google_id: str = ""
    google_secret: str = ""
    mollie_key: str = ""
    managed_kind: str = "openai"
    managed_url: str = ""
    managed_model: str = ""
    managed_key: str = ""
    installer: Path = Path("/app/downloads/AgentMilieu-Setup.exe")

    def __post_init__(self):
        url = urlsplit(self.public_url)
        if (not url.hostname or url.username or url.path or url.query or url.fragment
                or (url.scheme != "https" and
                    (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1")))):
            raise ValueError("PUBLIC_URL must be an HTTPS origin, or an HTTP localhost origin")
        if len(self.session_secret) < 32:
            raise ValueError("SESSION_SECRET must contain at least 32 characters")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")

    @property
    def secure(self):
        return self.public_url.startswith("https://")

    @property
    def managed(self):
        return bool(self.managed_model and self.managed_key)

    @classmethod
    def from_env(cls):
        get = os.environ.get
        return cls(
            database_url=get("DATABASE_URL", ""),
            public_url=get("PUBLIC_URL", "http://localhost:8080").rstrip("/"),
            session_secret=get("SESSION_SECRET", ""),
            google_id=get("GOOGLE_CLIENT_ID", ""),
            google_secret=get("GOOGLE_CLIENT_SECRET", ""),
            mollie_key=get("MOLLIE_API_KEY", ""),
            managed_kind=get("MANAGED_PROVIDER_KIND", "openai"),
            managed_url=get("MANAGED_PROVIDER_URL", ""),
            managed_model=get("MANAGED_PROVIDER_MODEL", ""),
            managed_key=get("MANAGED_PROVIDER_KEY", ""),
            installer=Path(get("WINDOWS_INSTALLER", "/app/downloads/AgentMilieu-Setup.exe")),
        )
