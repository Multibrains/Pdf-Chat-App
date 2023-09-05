from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Union


class Settings(BaseSettings):
    KEY: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
