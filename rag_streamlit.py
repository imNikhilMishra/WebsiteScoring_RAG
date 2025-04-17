import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pprint

# ============================
# CONFIGURATION (EDIT HERE)
# ============================
KB_PATH = "getgsi_knowledge_base.json"  # Path to your KB JSON file
GEMINI_API_KEY = "AIzaSyDfgN8TwKHmhoIH95oVXTFBWCJHU8wasFs"  # <<< Replace with your key
TOP_K = 5

# ============================
# INIT MODELS
# ============================
st.set_page_config(page_title="Website QA RAG", layout="wide")
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ============================
# LOAD KB + BUILD INDEX
# ============================
@st.cache_resource
def load_kb():
    with open(KB_PATH, "r") as f:
        kb = json.load(f)
    embeddings = np.array([entry["text_embedding"] for entry in kb]).astype("float32")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return kb, index

kb, index = load_kb()

# ============================
# STREAMLIT UI
# ============================
st.title("🔍 Website QA using RAG + Gemini")
st.markdown("Ask evaluative questions like:")
st.markdown("- Does the site have case studies?\n- Are testimonials shown?\n- Is the logo visible on the homepage?")
query = st.text_input("Enter your question:", placeholder="e.g., Are there any testimonials on the site?")

if query:
    # Embed query
    query_embedding = text_model.encode(query).astype("float32")

    # Retrieve top-k chunks
    D, I = index.search(np.array([query_embedding]), TOP_K)
    retrieved_chunks = [kb[i] for i in I[0]]

    # Print retrieved chunks to terminal
    print("\n========================")
    print(f"Query: {query}")
    print("\nTop Retrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nComponent {i+1}:")
        pprint.pprint({
            "css_selector": chunk["css_selector"],
            "text_content": chunk["text_content"][:300] + "...",
            "screenshot_path": chunk.get("screenshot_path", "N/A")
        })

    # Build prompt
    context_blocks = "\n\n".join(
        [f"Component {i+1} (Selector: {chunk['css_selector']}):\n{chunk['text_content']}"
         for i, chunk in enumerate(retrieved_chunks)]
    )
#     prompt = f"""
# You are a helpful assistant analyzing websites for Conversion Rate Optimization (CRO) signals, especially focusing on the presence and quality of **testimonials**.

# A testimonial is a quote, statement, or endorsement from a customer or client describing their positive experience with the business. It often includes:
# - The client's name or company
# - A quote or description of what was good about the product or service
# - Placement in carousels, quote blocks, or styled testimonial cards
# - Visual indicators like quotation marks, star ratings, or profile pictures

# **Important Instruction**: Do **not** hallucinate or assume the presence of a testimonial if it is not explicitly present in the retrieved content below. If the testimonial signal is not clearly found in the components provided, you **must respond** that the answer cannot be determined due to lack of relevant context or ineffective retrieval.

# **Question**: {query}

# **Components Extracted from the Website**:
# {context_blocks}

# Your task is to:
# 1. Determine if testimonials are present in any of the components of the provided website in the {query}.
# 2. If found, list the component(s) where they appear, and quote or summarize their content.
# 3. Judge how persuasive and clear they are (mention if names, companies, or quotes are missing).
# 4. If not found, explicitly say that the system could not find sufficient evidence for testimonials based on the retrieved context.

# Respond clearly and concisely.
# """

    prompt = f"""
You are a helpful assistant analyzing websites for Conversion Rate Optimization (CRO) signals, especially focusing on the presence and quality of **testimonials**.

A testimonial is a quote, statement, or endorsement from a customer or client describing their positive experience with the business. It often includes:
- The client's name or company
- A quote or description of what was good about the product or service
- Placement in carousels, quote blocks, or styled testimonial cards
- Visual indicators like quotation marks, star ratings, or profile pictures

**Important Instruction**: Try to be helpful and reasonable. If the components show hints or partial signals of testimonials (e.g., indirect quotes, phrases like "our customers say"), you may cautiously describe them as potential testimonials, while indicating the level of confidence.

However, **do not hallucinate or fabricate** testimonial details if they are not supported by any part of the provided content. If none of the components seem related to the testimonial signal at all, say that there isn’t enough relevant context to answer.

**Question**: {query}

**Components Extracted from the Website**:
{context_blocks}

Your task is to:
1. Determine if testimonial signals are present or partially suggested in the retrieved components.
2. If found or implied, describe them briefly and indicate how confident you are.
3. If not found at all, say clearly that the system couldn’t identify testimonial-related content in the retrieved chunks.

Respond clearly, be honest about confidence, and avoid speculation.
"""

    # Gemini LLM Call
    with st.spinner("Querying Gemini..."):
        response = gemini_model.generate_content(prompt)
        answer = response.text

    # Print prompt and answer to terminal
    print("\nPrompt Sent to Gemini:\n")
    print(prompt)
    print("\nAnswer from Gemini:\n")
    print(answer)

    # Output
    st.subheader("Answer")
    st.write(answer)

    # Display retrieved chunks in a useful format
    st.markdown("---")
    st.subheader("🔍 Retrieved Components")
    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Component {i+1} - Selector: {chunk['css_selector'][:60]}..."):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Text Content**")
                st.text_area("", chunk["text_content"], height=150)
                st.markdown("**HTML Snippet**")
                st.code(chunk["html"], language="html")
            with col2:
                if chunk.get("screenshot_path"):
                    st.image(chunk["screenshot_path"], caption="Screenshot", use_column_width=True)

    # Debugging Info
    st.markdown("---")
    with st.expander("⚙️ Debug: Distances to Retrieved Components"):
        for i, dist in enumerate(D[0]):
            st.markdown(f"Component {i+1}: Distance = {dist:.4f}")

    with st.expander("📄 Prompt Sent to Gemini"):
        st.code(prompt, language="markdown")
