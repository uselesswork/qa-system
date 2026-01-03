import re

class TextCleaner:

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 去除多余空白
        text = re.sub(r"\s+", " ", text)

        # 去除控制字符
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        return text.strip()
