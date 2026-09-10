from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from rank_bm25 import BM25Okapi
import re

class TrivialBaseline:
    """
    Trivial Baseline Model:
    - Intent: Predicts majority class ('ACCOUNT_BILLING')
    - Reply: Static boilerplate response
    - Action: Always AUTO_HANDLE
    """
    def __init__(self):
        self.majority_intent = "ACCOUNT_BILLING"
        
    def process_message(self, message: str) -> dict:
        return {
            "inbound_message": message,
            "predicted_intent": self.majority_intent,
            "intent_confidence": 0.50,
            "drafted_reply": "Thank you for contacting Apple Support. Please send us a DM with your Apple ID email for assistance.",
            "action": "AUTO_HANDLE",
            "escalation_reason": "Trivial Baseline: Always auto-handles by default."
        }


class SimpleBaseline:
    """
    Simple Baseline Model:
    - Intent: Basic TF-IDF Vectorizer + Multinomial Naive Bayes
    - Reply: Raw BM25 top search result without RAG template formatting
    - Action: Simple keyword heuristic ('hacked', 'refund', 'broken' -> ESCALATE, else AUTO_HANDLE)
    """
    def __init__(self, training_df, knowledge_base):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.pipeline.fit(training_df['clean_inbound'], training_df['intent'])
        
        self.knowledge_base = knowledge_base
        corpus = [re.findall(r'\w+', item['historical_customer_query'].lower()) for item in knowledge_base]
        self.bm25 = BM25Okapi(corpus)

    def process_message(self, message: str) -> dict:
        # Intent prediction
        pred_intent = self.pipeline.predict([message.lower()])[0]
        probs = self.pipeline.predict_proba([message.lower()])[0]
        confidence = float(max(probs))

        # BM25 Search for raw reply
        tokens = re.findall(r'\w+', message.lower())
        scores = self.bm25.get_scores(tokens)
        best_idx = int(scores.argmax()) if len(scores) > 0 else 0
        raw_reply = self.knowledge_base[best_idx]['brand_resolution']

        # Simple heuristic escalation
        trigger_words = ["hacked", "stolen", "refund", "broken", "court", "suck"]
        if any(w in message.lower() for w in trigger_words):
            action = "ESCALATE"
            reason = "Simple Baseline Heuristic: Contains trigger keyword."
        else:
            action = "AUTO_HANDLE"
            reason = "Simple Baseline Heuristic: No trigger keyword."

        return {
            "inbound_message": message,
            "predicted_intent": pred_intent,
            "intent_confidence": round(confidence, 4),
            "drafted_reply": raw_reply,
            "action": action,
            "escalation_reason": reason
        }
