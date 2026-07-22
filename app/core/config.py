from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    app_env: str = "development"
    checkpoint_path: str = "checkpoints/best_model.pth"
    num_classes: int = 3
    spatial_size: int = 128
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    log_level: str = "info"
    database_url: str = "sqlite+aiosqlite:///./neuropredict.db"
    database_echo: bool = False

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "protected_namespaces": ()}


settings = Settings()
