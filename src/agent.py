from src.dataset_loader import DatasetLoader
from src.intent_classifier import IntentClassifier
from src.rag_engine import RAGEngine
from src.escalation_engine import EscalationEngine

class SupportAgent:
    """
    Unified AI Customer Support Agent for Twitter.
    Orchestrates Intent Classification -> Grounded RAG Retrieval -> Escalation Decision.
    """

    def __init__(self, data_dir="data"):
        self.loader = DatasetLoader(data_dir=data_dir)
        
        # Load datasets
        self.kb = self.loader.load_historical_knowledge()
        self.raw_df = self.loader.load_raw_samples()
        
        # Initialize components
        self.classifier = IntentClassifier()
        self.rag = RAGEngine(knowledge_base=self.kb)
        self.escalator = EscalationEngine()
        
        # Train intent classifier on historical samples
        self._fit_classifier()

    def _fit_classifier(self):
        """Fit the internal intent classifier using raw historical tweets."""
        texts = self.raw_df['clean_inbound'].tolist()
        labels = self.raw_df['intent'].tolist()
        self.classifier.train(texts, labels)

    def process_message(self, customer_message: str) -> dict:
        """
        Process an incoming customer tweet through the 3-step pipeline:
        1. Classify intent
        2. Retrieve grounded resolution & draft reply
        3. Evaluate auto-handle vs escalation with reason
        """
        # Step 1: Classify Intent
        intent_res = self.classifier.predict(customer_message)
        predicted_intent = intent_res['intent']

        # Step 2: Grounded RAG Retrieval
        retrieved_docs = self.rag.retrieve(customer_message, intent=predicted_intent, top_k=1)
        top_rag = retrieved_docs[0] if retrieved_docs else None
        
        draft_reply = self.rag.draft_reply(customer_message, predicted_intent, retrieved_docs)

        # Step 3: Escalation Decision
        escalation_res = self.escalator.evaluate(customer_message, intent_res, top_rag)

        # Build final unified decision payload
        return {
            "inbound_message": customer_message,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_res['confidence'],
            "grounded_knowledge_id": top_rag['kb_id'] if top_rag else None,
            "grounded_similarity_score": top_rag['similarity_score'] if top_rag else 0.0,
            "drafted_reply": draft_reply,
            "action": escalation_res['action'],  # AUTO_HANDLE or ESCALATE
            "escalation_reason": escalation_res['reason'],
            "risk_score": escalation_res['risk_score'],
            "escalation_trigger": escalation_res['trigger']
        }
