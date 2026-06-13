from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "medibot_chunks"
    data_dir: Path = Path("../mediassist_data")
    sqlite_db_path: Path = Path("../mediassist_data/db/mediassist.db")
    allow_local_llm_fallback: bool = True
    enable_local_ml_models: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
