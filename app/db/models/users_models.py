import datetime as dt
import enum
import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship

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
    """Модель активности пользователей"""

    __tablename__ = 'user_activity'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(User.id), index=True)

    user: Mapped[User] = relationship()

    request_url: Mapped[str] = mapped_column(nullable=True, default=None)
    request_type: Mapped[str] = mapped_column(nullable=True, default=None)
    request_time: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
