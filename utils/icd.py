from collections.abc import Sequence

import toml


class ICD:
    """A ICD instance represents ICD information of disease.

    It must be hashable so that can be used in functools.cache.
    """

    def __init__(self, v10: Sequence[str], v9: Sequence[str]) -> None:
        if (len(v10) != len(set(v10))) or (len(v9) != len(set(v9))):
            raise ValueError("duplicate code")
        self.v10 = tuple(v10)
        self.v9 = tuple(v9)


def load_icd(disease: str) -> ICD:
    with open("../configs/icd.toml", encoding="utf16") as f:
        d = toml.load(f)[disease]
    return ICD(v10=d["10"], v9=d["9"])
