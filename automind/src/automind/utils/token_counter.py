import json

import pandas as pd
import tiktoken


class TokenCounter:
    def __init__(self, model_encoding="cl100k_base"):
        self.encoding = tiktoken.get_encoding(model_encoding)

    def count_text(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_dataframe_raw(self, df: pd.DataFrame) -> int:
        csv_string = df.to_csv(index=False)
        return self.count_text(csv_string)

    def count_metadata(self, metadata: dict) -> int:
        json_str = json.dumps(metadata, ensure_ascii=False)
        return self.count_text(json_str)
