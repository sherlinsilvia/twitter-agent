import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class IntentClassifier:
    """
    Classifies incoming customer support messages into core intent categories.
    Combines TF-IDF Machine Learning classification with rule-based keyword signals.
    """
    
    INTENTS = [
        "ACCOUNT_BILLING",
        "TECHNICAL_HARDWARE",
        "ORDER_SHIPPING",
        "REFUND_CANCEL",
        "SERVICE_OUTAGE",
        "GENERAL_FEEDBACK"
    ]
    
    KEYWORD_SIGNALS = {
        "ACCOUNT_BILLING": ["charge", "charged", "billing", "invoice", "payment", "card", "unauthorized", "subscription", "icloud storage", "apple id", "declined"],
        "TECHNICAL_HARDWARE": ["battery", "drain", "screen", "black", "frozen", "overheat", "airpods", "crackling", "macbook", "fan", "bluetooth", "face id", "camera", "ios"],
        "ORDER_SHIPPING": ["order", "shipping", "fedex", "ups", "tracking", "delivered", "package", "porch", "trade-in", "courier", "dispatch", "pickup"],
        "REFUND_CANCEL": ["refund", "cancel", "cancellation", "money back", "return", "purchase", "in-app", "store credit", "appeal"],
        "SERVICE_OUTAGE": ["down", "outage", "system status", "server", "imessage", "app store", "apple music", "sync", "connection", "cannot connect", "buffering"],
        "GENERAL_FEEDBACK": ["feedback", "love", "shoutout", "kudos", "feature request", "great", "thanks", "thank you", "store hours", "guide", "session"]
    }

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.is_fitted = False

    def train(self, texts, labels):
        """Train the TF-IDF intent classifier model."""
        texts_list = list(texts)
        labels_list = list(labels)
        if len(texts_list) == 0 or len(labels_list) == 0:
            raise ValueError("Training dataset cannot be empty.")
        self.pipeline.fit(texts_list, labels_list)
        self.is_fitted = True

    def predict(self, text: str) -> dict:
        """
        Predict intent for a given input tweet.
        Returns dictionary with predicted intent, confidence score, and top alternative.
        """
        text_clean = text.lower().strip()
        
        # ML Prediction if fitted
        if self.is_fitted:
            probs = self.pipeline.predict_proba([text_clean])[0]
            classes = self.pipeline.classes_
            best_idx = np.argmax(probs)
            pred_intent = classes[best_idx]
            confidence = float(probs[best_idx])
        else:
            pred_intent, confidence = self._rule_based_predict(text_clean)

        # Keyword Boost for high precision on edge cases
        rule_intent, rule_score = self._rule_based_predict(text_clean)
        if rule_score >= 2 and rule_intent != pred_intent:
            # Overwrite if rule signal is very strong
            pred_intent = rule_intent
            confidence = max(confidence, 0.85)

        return {
            "intent": pred_intent,
            "confidence": round(confidence, 4),
            "method": "hybrid_tfidf_keyword"
        }

    def _rule_based_predict(self, text: str) -> tuple:
        """Fallback keyword scoring for intent prediction."""
        scores = {}
        for intent, keywords in self.KEYWORD_SIGNALS.items():
            matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text))
            scores[intent] = matches
        
        best_intent = max(scores, key=scores.get)
        match_count = scores[best_intent]
        
        if match_count == 0:
            return "GENERAL_FEEDBACK", 0.33
        
        confidence = min(0.5 + (match_count * 0.15), 0.95)
        return best_intent, confidence
