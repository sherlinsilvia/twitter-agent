import streamlit as st
import sys
import os

# Ensure src module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import SupportAgent
from src.dataset_loader import DatasetLoader

st.set_page_config(
    page_title="AI Support Agent - Twitter Brand Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Twitty Agent")
st.caption("End-to-End Intent Classification, Grounded RAG Retrieval, and Explainable Escalation Engine")

@st.cache_resource
def load_agent():
    return SupportAgent(data_dir="data")

agent = load_agent()

# Sidebar options
st.sidebar.header("🔧 Configuration & Controls")
sample_queries = [
    "I was charged $2.99 twice for my iCloud storage subscription this month!",
    "Someone hacked into my Apple ID and changed my credit card details!",
    "My iPhone 13 battery drains from 100% to zero in 2 hours after updating iOS.",
    "Tracking says package delivered today but I checked my porch and nothing is there!",
    "Where is my order #W88492018? FedEx tracking has been stuck for 4 days.",
    "Is App Store down right now? 'Cannot Connect' error on all my family devices.",
    "I love the new camera cinematic mode on iPhone! Great work design team!"
]

selected_sample = st.sidebar.selectbox("Select Sample Customer Tweet:", ["Custom Input..."] + sample_queries)

if selected_sample != "Custom Input...":
    user_input = selected_sample
else:
    user_input = st.text_area(
        "Enter Incoming Customer Tweet:",
        "Hi @AppleSupport, I was charged twice for my subscription this month. Can I get a refund?",
        height=100
    )

if st.button("🚀 Process Message with AI Agent", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Processing through Intent Classifier, RAG Index, and Escalation Guardrails..."):
            result = agent.process_message(user_input)
            
        st.markdown("---")
        
        # Grid display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Intent Classification")
            st.metric("Predicted Intent", result['predicted_intent'])
            st.metric("Confidence Score", f"{result['intent_confidence']:.2%}")
            
        with col2:
            st.subheader("2. Grounded RAG Retrieval")
            st.metric("Knowledge Article", result['grounded_knowledge_id'] or "None")
            st.metric("Relevance Match Score", f"{result['grounded_similarity_score']:.2f}")

        with col3:
            st.subheader("3. Action & Escalation Decision")
            action = result['action']
            if action == "AUTO_HANDLE":
                st.success(f"🟢 **{action}**")
            else:
                st.error(f"🚨 **{action}**")
            st.metric("Risk Score", f"{result['risk_score']:.2f}")

        st.markdown("### 📝 Decision & Stated Rationale")
        st.info(f"**Reason:** {result['escalation_reason']}")

        st.markdown("### 💬 Drafted Brand Response (Grounded)")
        st.code(result['drafted_reply'], language="text")

st.markdown("---")