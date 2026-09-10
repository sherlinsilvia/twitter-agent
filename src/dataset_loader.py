import os
import json
import re
import pandas as pd

class DatasetLoader:
    """
    Utility class for loading and preprocessing dataset files for the Twitter AI Support Agent.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize customer tweet text."""
        if not text or pd.isna(text):
            return ""
        # Lowercase and clean spacing
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_raw_samples(self, filename="raw_sample.csv") -> pd.DataFrame:
        """Load customer support tweet samples."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Raw dataset file not found at {filepath}")
        
        df = pd.read_csv(filepath)
        df['clean_inbound'] = df['inbound_text'].apply(self.clean_text)
        return df

    def load_historical_knowledge(self, filename="historical_knowledge.json") -> list:
        """Load curated brand resolution knowledge base."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Knowledge base file not found at {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_golden_eval_set(self, filename="golden_eval_set.json") -> list:
        """Load hand-labelled golden evaluation dataset."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Golden evaluation set not found at {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        return eval_set
