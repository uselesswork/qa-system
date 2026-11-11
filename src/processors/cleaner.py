import re

class TextCleaner:

    def clean(self, text: str, preserve_newlines: bool = False) -> str:
        if not text:
            return ""

        if preserve_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"[^\S\n]+", " ", text)
            text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
            return text.strip()

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)
        return text.strip()
