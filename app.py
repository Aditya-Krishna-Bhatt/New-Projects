import streamlit as st
import os
import pypdf
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. STREAMLIT INTERFACE SETUP
st.set_page_config(page_title="100% Free RAG Assistant", layout="wide")
st.title("🤖 Free Intelligent Document Assistant (Native Semantic Window)")
st.write("Upload a PDF document and query it contextually within free token limits.")

# Sidebar Configuration Panel
api_key = st.sidebar.text_input("Enter your Free Groq API Key (starts with gsk_)", type="password")
uploaded_file = st.file_uploader("Upload your reference PDF", type="pdf")

# Use Session State memory arrays to prevent infinite reload loop triggers
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None

if uploaded_file and api_key:
    if st.button("🚀 Process and Parse Document"):
        try:
            with st.spinner("⏳ Step 1: Extracting raw text and chunking document layers..."):
                pdf_reader = pypdf.PdfReader(uploaded_file)
                chunks = []
                
                # Parse page by page to keep structural contexts isolated
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        # Clean up irregular line breaks to optimize reading layout
                        cleaned_text = re.sub(r'\s+', ' ', text).strip()
                        # Split page text into manageable 800-character blocks
                        for i in range(0, len(cleaned_text), 800):
                            chunk_snippet = cleaned_text[i:i+1000] # 200 char overlap window
                            chunks.append({"page": page_num + 1, "text": chunk_snippet})
                
                st.session_state.pdf_chunks = chunks
                
            if st.session_state.pdf_chunks:
                st.success(f"✅ Ingestion complete! Fragmented text into {len(st.session_state.pdf_chunks)} semantic context blocks.")
            else:
                st.error("⚠️ Failed to parse text layers. Verify that the PDF is not an un-scanned flat image binary.")
                
        except Exception as e:
            st.error(f"Ingestion Failure: {e}")

# 2. LIGHTWEIGHT KEYWORD SIMILARITY SEARCH ENGINE
def retrieve_relevant_context(query, chunks, top_k=2):
    # Extracts words from the question to use as scoring anchors
    query_words = [word.lower() for word in re.findall(r'\w+', query) if len(word) > 3]
    scored_chunks = []
    
    for chunk in chunks:
        score = 0
        chunk_text_lower = chunk["text"].lower()
        for word in query_words:
            if word in chunk_text_lower:
                score += 1
        if score > 0:
            scored_chunks.append((score, chunk))
            
    # Sort chunks by highest match score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Fallback if no matching keywords are found: take the first two chunks
    if not scored_chunks:
        return "\n\n".join([c["text"] for c in chunks[:top_k]])
        
    # Extract text from top-scoring chunks
    top_matches = [item[1]["text"] for item in scored_chunks[:top_k]]
    return "\n\n".join(top_matches)

# 3. CONVERSATIONAL QUEUE INTERFACE
if st.session_state.pdf_chunks is not None:
    user_query = st.text_input("💬 Ask a specific question about your document:")
    
    if user_query:
        with st.spinner("🔍 Filtering semantic blocks and routing through Groq..."):
            try:
                # Isolate only the relevant pages matching the query keywords
                filtered_context = retrieve_relevant_context(user_query, st.session_state.pdf_chunks, top_k=2)
                
                # Execute inference using Groq's active production cluster
                os.environ["GROQ_API_KEY"] = api_key
                llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.1)
                
                # Configure custom prompt template to enforce strict factual safety constraints
                system_prompt = (
                    "You are an expert analytical assistant. Answer the user query based strictly "
                    "on the provided context below. If the provided context does not contain "
                    "the details to answer, state clearly that the text does not offer enough data.\n\n"
                    f"Context:\n{filtered_context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # Process pipeline natively
                formatted_messages = prompt.format_messages(input=user_query)
                response = llm.invoke(formatted_messages)
                
                st.write("### 📢 Assistant Response:")
                st.info(response.content)
                
            except Exception as e:
                st.error(f"Inference Error during processing: {e}")
                
elif not api_key:
    st.warning("⚠️ Please provide a valid Groq API key in the sidebar configuration panel.")
