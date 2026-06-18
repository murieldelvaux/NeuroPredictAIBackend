from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    app_env: str = "development"
    model_checkpoint_path: str = "checkpoints/best_model.pth"
    model_num_classes: int = 3
    model_spatial_size: int = 128
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    log_level: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
