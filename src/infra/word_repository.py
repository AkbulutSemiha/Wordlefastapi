import os
import pandas as pd
from typing import List

class WordRepository:
    def __init__(self, data_dir: str = "Words"):
        self.data_dir = data_dir
        self.words_tr = []
        self.words_en = []
        self._load_all()

    def _load_all(self):
        tr_path = os.path.join(self.data_dir, "words_tr.txt")
        en_path = os.path.join(self.data_dir, "words_en.txt") # Assuming this exists or will be added
        
        self.words_tr = self._read_file(tr_path)
        if os.path.exists(en_path):
            self.words_en = self._read_file(en_path)

    def _read_file(self, path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().upper() for line in f if line.strip()]

    def get_words(self, language: str = "tr") -> List[str]:
        return self.words_tr if language == "tr" else self.words_en

    def get_frequent_words(self, language: str = "tr", limit: int = 1000) -> List[str]:
        """
        Loads frequent words based on the provided CSV file.
        This is useful for simulation experiments.
        """
        csv_path = os.path.join(self.data_dir, "five_letter_word_frequencies.csv")
        if not os.path.exists(csv_path):
            return self.get_words(language)[:limit]
            
        df = pd.read_csv(csv_path)
        alpha = 0.5
        df["freq_norm"] = df["frequency"].rank(pct=True)
        df["score"] = df["freq_norm"] * alpha + df["dispersion"] * (1 - alpha)
        
        tr_map = str.maketrans("çğiöşü", "ÇĞİÖŞÜ")
        df["word"] = df["word"].apply(lambda x: x.translate(tr_map).upper())
        
        all_words = self.get_words(language)
        filtered = df[df["word"].isin(all_words)]
        return filtered.sort_values("score", ascending=False).head(limit)["word"].tolist()
