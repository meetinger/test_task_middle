import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent


class Settings(BaseSettings):
    """Класс конфигурации, считывается из .env"""

    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str

    HTTP_PORT: int

    model_config = SettingsConfigDict(env_file=ROOT_DIR.joinpath('.env'))

    def get_db_url(self, driver: str = 'asyncpg'):
        """Получение ссылки на БД"""
        return f'postgresql+{driver}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'


settings = Settings()
