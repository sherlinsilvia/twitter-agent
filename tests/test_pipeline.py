import unittest
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import DatasetLoader
from src.intent_classifier import IntentClassifier
from src.rag_engine import RAGEngine
from src.escalation_engine import EscalationEngine
from src.agent import SupportAgent
from src.evaluator import Evaluator

class TestSupportAgentPipeline(unittest.TestCase):

    def test_dataset_loader(self):
        loader = DatasetLoader(data_dir="data")
        raw_df = loader.load_raw_samples()
        kb = loader.load_historical_knowledge()
        golden_eval = loader.load_golden_eval_set()
        
        self.assertGreater(len(raw_df), 0)
        self.assertGreater(len(kb), 0)
        self.assertGreater(len(golden_eval), 0)

    def test_intent_classifier(self):
        loader = DatasetLoader(data_dir="data")
        raw_df = loader.load_raw_samples()
        
        clf = IntentClassifier()
        clf.train(raw_df['clean_inbound'], raw_df['intent'])
        
        res = clf.predict("I was charged twice for my subscription!")
        self.assertIn(res['intent'], IntentClassifier.INTENTS)
        self.assertGreater(res['confidence'], 0.0)

    def test_rag_engine(self):
        loader = DatasetLoader(data_dir="data")
        kb = loader.load_historical_knowledge()
        
        rag = RAGEngine(kb)
        results = rag.retrieve("My iPhone battery is draining super fast", intent="TECHNICAL_HARDWARE")
        
        self.assertEqual(len(results), 1)
        self.assertIn("resolution_steps", results[0])
        
        reply = rag.draft_reply("Battery issue", "TECHNICAL_HARDWARE", results)
        self.assertGreater(len(reply), 10)

    def test_escalation_engine(self):
        escalator = EscalationEngine()
        
        # Safe query test
        res1 = escalator.evaluate(
            "How do I track my online order?",
            {"intent": "ORDER_SHIPPING", "confidence": 0.90},
            {"similarity_score": 0.85}
        )
        self.assertEqual(res1['action'], "AUTO_HANDLE")
        
        # Security escalation test
        res2 = escalator.evaluate(
            "Someone hacked my Apple ID account!",
            {"intent": "ACCOUNT_BILLING", "confidence": 0.90},
            {"similarity_score": 0.85}
        )
        self.assertEqual(res2['action'], "ESCALATE")
        self.assertIn("Security", res2['reason'])

    def test_full_agent_pipeline(self):
        agent = SupportAgent(data_dir="data")
        res = agent.process_message("I was charged twice for iCloud storage.")
        
        self.assertIn("predicted_intent", res)
        self.assertIn("drafted_reply", res)
        self.assertIn(res['action'], ["AUTO_HANDLE", "ESCALATE"])
        self.assertIn("escalation_reason", res)

    def test_evaluator(self):
        agent = SupportAgent(data_dir="data")
        golden_eval_set = agent.loader.load_golden_eval_set()[:10]  # sample of 10
        
        evaluator = Evaluator()
        results = evaluator.evaluate_pipeline(agent, golden_eval_set)
        
        self.assertEqual(results['total_samples'], 10)
        self.assertTrue(0.0 <= results['intent_classification']['accuracy'] <= 1.0)
        self.assertTrue(1.0 <= results['reply_quality']['llm_judge_avg_score'] <= 5.0)

if __name__ == "__main__":
    unittest.main()
