import re
from rank_bm25 import BM25Okapi

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) Engine for Twitter Support.
    Retrieves grounded historical brand resolutions based on intent and semantic/keyword similarity.
    """
    
    def __init__(self, knowledge_base: list):
        self.knowledge_base = knowledge_base
        self.bm25 = None
        self.corpus = []
        self._build_index()

    def _tokenize(self, text: str) -> list:
        """Tokenize text into lowercased alphanumeric words."""
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        """Index knowledge base items with Rank-BM25."""
        self.corpus = []
        for item in self.knowledge_base:
            combined_text = f"{item['intent']} {' '.join(item.get('issue_keywords', []))} {item['historical_customer_query']} {item['brand_resolution']}"
            self.corpus.append(self._tokenize(combined_text))
        
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, intent: str = None, top_k: int = 1) -> list:
        """
        Retrieve top-k grounded historical resolutions for a query.
        Returns list of matching knowledge base items with similarity score.
        """
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Rank items
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        results = []
        for idx in ranked_indices:
            item = self.knowledge_base[idx]
            raw_score = float(bm25_scores[idx])
            
            # Boost score if intent matches
            intent_boost = 1.5 if intent and item['intent'] == intent else 1.0
            final_score = min(raw_score * intent_boost / 10.0, 1.0)  # Normalize between 0 and 1
            
            results.append({
                "kb_id": item['id'],
                "intent": item['intent'],
                "query": item['historical_customer_query'],
                "brand_resolution": item['brand_resolution'],
                "resolution_steps": item.get('resolution_steps', []),
                "similarity_score": round(final_score, 4)
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def draft_reply(self, query: str, intent: str, retrieved_docs: list) -> str:
        """
        Draft a brand response grounded in historical resolutions.
        In offline mode, formats the retrieved brand resolution template into a friendly tweet response.
        """
        if not retrieved_docs:
            return "We are here to help! Please send us a DM with your Apple ID email and issue details so our support team can assist you."

        top_doc = retrieved_docs[0]
        base_resolution = top_doc['brand_resolution']
        
        # Customize greeting & tone for Twitter
        if "DM" in base_resolution:
            reply = f"Hi there! {base_resolution}"
        else:
            reply = f"Thanks for reaching out! {base_resolution} DM us if you need further help."
            
        return reply
