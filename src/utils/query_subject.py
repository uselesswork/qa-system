"""可确定识别的历史朝代限定词，用于防止改写或证据串题。"""

import re


_DYNASTY_MENTIONS = re.compile(
    r"(西汉|东汉|北宋|南宋|西晋|东晋|夏|商|周|秦|汉|晋|隋|唐|宋|辽|金|元|明|清)(?:朝|代)"
)


def mentioned_dynasties(text: str) -> set[str]:
    return set(_DYNASTY_MENTIONS.findall(text))
