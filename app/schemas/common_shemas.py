from typing import TypeVar, Generic

from pydantic import BaseModel, Field

Entity = TypeVar('Entity')  # дженерик для сущностей


class EntityList(BaseModel, Generic[Entity]):
    """Схема для списка сущностей"""
    entities: list[Entity] = Field(description='Список сущностей')
