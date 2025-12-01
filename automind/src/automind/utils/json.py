import re
from typing import Protocol


class JsonCleanerProtocal(Protocol):
    @staticmethod
    def clean_json_text(json_text: str) -> str: ...


class JsonCleaner:
    @staticmethod
    def clean_json_text(json_text: str) -> str:
        """
        Clean up malformed JSON text by removing common formatting issues.

        Args:
            json_text: Potentially malformed JSON string

        Returns:
            Cleaned JSON string
        """
        # extract JSON content between first { and last }
        start_idx = json_text.find("{")
        end_idx = json_text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_text = json_text[start_idx : end_idx + 1]

        # remove trailing commas
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)

        return json_text
