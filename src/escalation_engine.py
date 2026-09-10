import re

class EscalationEngine:
    """
    Decides whether an incoming customer tweet should be automatically handled by the AI agent
    or escalated to a human customer support agent, accompanied by an explicit, auditable reason.
    """

    ESCALATION_KEYWORDS = {
        "SECURITY_ALERT": ["hacked", "stolen", "compromised", "unauthorized", "phishing", "scam", "fraud"],
        "HARDWARE_DAMAGE": ["dropped", "broken screen", "water damage", "smoke", "fire", "exploded", "melted"],
        "SENSITIVE_FINANCIAL": ["child spent", "unauthorized charge", "lawsuit", "legal", "stolen card", "bank refund"],
        "HIGH_FRUSTRATION": ["terrible", "worst", "suck", "scam", "useless", "lawyer", "report to fcc", "furious", "disgusted"]
    }

    CONFIDENCE_THRESHOLD = 0.50

    def evaluate(self, query: str, intent_info: dict, rag_info: dict) -> dict:
        """
        Evaluates the message and returns an escalation decision:
        - action: "AUTO_HANDLE" or "ESCALATE"
        - reason: Transparent human-readable explanation
        - risk_score: Numeric score from 0.0 (low risk) to 1.0 (high risk)
        """
        query_lower = query.lower()
        confidence = intent_info.get("confidence", 1.0)
        retrieval_score = rag_info.get("similarity_score", 1.0) if rag_info else 0.0

        # Trigger 1: Security or Account Compromise
        for category, kws in self.ESCALATION_KEYWORDS.items():
            for kw in kws:
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                    if category == "SECURITY_ALERT":
                        return {
                            "action": "ESCALATE",
                            "reason": f"Account Security Trigger: Detected key security phrase ('{kw}'). Requires human security review.",
                            "risk_score": 0.95,
                            "trigger": category
                        }
                    elif category == "HARDWARE_DAMAGE":
                        return {
                            "action": "ESCALATE",
                            "reason": f"Physical Damage Trigger: Detected potential hardware damage ('{kw}'). Requires human technician diagnostic.",
                            "risk_score": 0.85,
                            "trigger": category
                        }
                    elif category == "SENSITIVE_FINANCIAL":
                        return {
                            "action": "ESCALATE",
                            "reason": f"Financial Liability Trigger: Detected sensitive billing phrase ('{kw}'). Requires human supervisor approval.",
                            "risk_score": 0.90,
                            "trigger": category
                        }
                    elif category == "HIGH_FRUSTRATION":
                        return {
                            "action": "ESCALATE",
                            "reason": f"Customer Sentiment Trigger: High frustration or legal threat detected ('{kw}'). Escalating for empathetic human care.",
                            "risk_score": 0.80,
                            "trigger": category
                        }

        # Trigger 2: Low Classifier Confidence
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "action": "ESCALATE",
                "reason": f"Low Intent Confidence ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD:.2f}). Message intent is ambiguous.",
                "risk_score": 0.65,
                "trigger": "LOW_INTENT_CONFIDENCE"
            }

        # Trigger 3: Low Retrieval Confidence
        if retrieval_score < 0.20:
            return {
                "action": "ESCALATE",
                "reason": f"Low Grounding Confidence ({retrieval_score:.2f} < 0.20). No close historical resolution match in knowledge base.",
                "risk_score": 0.60,
                "trigger": "LOW_RAG_CONFIDENCE"
            }

        # Default: Safe for Auto-Handling
        return {
            "action": "AUTO_HANDLE",
            "reason": f"High confidence intent ({intent_info.get('intent')}, score: {confidence:.2f}) and grounded historical resolution match.",
            "risk_score": 0.15,
            "trigger": "NONE"
        }
