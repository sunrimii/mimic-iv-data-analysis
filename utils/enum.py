from __future__ import annotations

from enum import Enum


class StreamlitEnum(str, Enum):
    def __eq__(self, other) -> bool:
        """Enable to check equality by value."""
        if not isinstance(other, StreamlitEnum):
            return NotImplemented
        return self.value == other.value

    @classmethod
    def to_list(cls) -> list[str]:
        return [e.value for e in cls]


class Gender(StreamlitEnum):
    ALL = "all"
    MALE = "male"
    FEMALE = "female"
