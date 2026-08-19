import streamlit as st

import config
from backend import generate_response

st.set_page_config(
    page_title="OsteoGuard AI",
    page_icon="🦴",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: transparent;
        padding: 10px 0;
    }
    div[data-testid="metric-container"] > div {
        color: #475569;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Footer styling */
    .footer {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    
    /* Button styles */
    .example-btn-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .stButton>button[kind="secondary"] {
        border-color: #cbd5e1;
        color: #475569;
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
    }
    .stButton>button[kind="secondary"]:hover {
        border-color: #94a3b8;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'query' not in st.session_state:
    st.session_state.query = ""

def set_query(q):
    st.session_state.query = q

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🦴 OsteoGuard AI")
    st.markdown("**Evidence-grounded clinical decision support**")
    st.caption("OsteoGuard retrieves relevant guideline evidence and generates responses grounded only in the retrieved material.")
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    # Credentials come from deployment configuration, not from this interface.
    configured = config.api_key_configured()
            
    num_chunks = st.slider("Number of evidence chunks", min_value=1, max_value=10, value=5)
    st.divider()
    
    st.markdown("### 🔒 Safety")
    st.caption("This system is intended to support clinicians. It does not replace professional medical judgment.")
    
    if not configured:
        st.warning(config.MISSING_KEY_MESSAGE)

# --- MAIN CONTENT ---
st.markdown("<h1>🦴 OsteoGuard AI</h1>", unsafe_allow_html=True)
st.markdown("### Evidence-Grounded Clinical Decision Support for Osteoarthritis")
st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Evidence First", "Guideline-Based")
col2.metric("Citation", "Every Claim")
col3.metric("Safety", "Clinician-Centered")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔎 Ask a Clinical Question")

# Input area
user_query = st.text_area(
    "Clinical Question Input",
    value=st.session_state.query,
    placeholder="Example: What is the clinical role of duloxetine in managing osteoarthritis?",
    label_visibility="collapsed"
)

st.markdown("**Example questions:**")
btn_col1, btn_col2, btn_col3 = st.columns(3)
if btn_col1.button("What is the clinical role of duloxetine in managing osteoarthritis?", use_container_width=True):
    set_query("What is the clinical role of duloxetine in managing osteoarthritis?")
    st.rerun()
if btn_col2.button("What non-pharmacological interventions are recommended for osteoarthritis?", use_container_width=True):
    set_query("What non-pharmacological interventions are recommended for osteoarthritis?")
    st.rerun()
if btn_col3.button("What does the guideline recommend for exercise in knee osteoarthritis?", use_container_width=True):
    set_query("What does the guideline recommend for exercise in knee osteoarthritis?")
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
submit_clicked = st.button("⚡ Retrieve Evidence & Generate Answer", type="primary", use_container_width=True)

# Generate response
if submit_clicked:
    if not user_query.strip():
        st.warning("Please enter a question first.")
    elif not configured:
        st.error(config.MISSING_KEY_MESSAGE)
    else:
        st.markdown("### 📊 Response")
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                response_text, sources = generate_response(user_query, top_n=num_chunks)
                
                # Display Answer
                st.markdown(response_text)
                
                # Display Sources
                if sources:
                    st.markdown("---")
                    st.markdown("#### 📚 Retrieved Evidence")
                    for idx, src in enumerate(sources, 1):
                        with st.expander(f"[{idx}] {src['doc_name']} (Page {src['page']}) - Relevance: {src['score']:.2f}"):
                            st.markdown(f"[Source Link]({src['url']})")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("#### 🦴 OsteoGuard AI")
st.caption("Evidence-grounded clinical decision support for osteoarthritis.")
st.caption("Always verify recommendations against the current clinical guideline and use professional clinical judgment.")
st.markdown("</div>", unsafe_allow_html=True)
