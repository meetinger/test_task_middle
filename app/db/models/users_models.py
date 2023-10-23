import datetime as dt
import uuid

from sqlalchemy.orm import mapped_column, Mapped

from app.db.database import Base


class User(Base):
    """ORM модель пользователя
    Время в БД лучше хранить в UTC, чтобы не было проблем с часовыми поясами и переводами времени на зимнее/летнее
    Используем naive datetime, другие asyncpg не принимает

    """

    __tablename__ = 'user'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(index=True, unique=True)
    email: Mapped[str] = mapped_column(index=True, unique=True)
    password_hash: Mapped[str] = mapped_column()
    registration_date: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class UserActivity(Base):
    __tablename__ = 'user_activity'

    id: Mapped[int] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
