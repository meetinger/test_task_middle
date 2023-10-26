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

    DB_TEST_USER: str
    DB_TEST_PASSWORD: str
    DB_TEST_HOST: str
    DB_TEST_PORT: str
    DB_TEST_NAME: str

    HTTP_PORT: int

    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_MINUTES: int
    SECRET_KEY: str
    ALGORITHM: str

    model_config = SettingsConfigDict(env_file=ROOT_DIR.joinpath('.env'))

    def get_db_url(self, driver: str = 'asyncpg', test: bool = False):
        """Получение ссылки на БД"""
        if test:
            return (f'postgresql+{driver}://{self.DB_TEST_USER}:{self.DB_TEST_PASSWORD}'
                    f'@{self.DB_TEST_HOST}:{self.DB_TEST_PORT}/{self.DB_TEST_NAME}')
        return f'postgresql+{driver}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'


settings = Settings()
