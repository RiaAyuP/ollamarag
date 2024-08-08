import streamlit as st
import ollama
import base64

st.title("📝 File Q&A with Ollama")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
model = st.selectbox("Please select LLM",("llama3.1", "gemma2", "mistral"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not (uploaded_file or uploaded_pdf),
)

if uploaded_pdf and uploaded_file:
    st.warning("Please upload one file only.", icon="⚠️")

if question:
    if uploaded_file:
        article = uploaded_file.read().decode()
    else:
        article = base64.b64encode(uploaded_pdf.read()).decode("utf-8")
    
    prompt = f"""<bos><start_of_turn>user\nAnswer the question based only on the following context \
        and extract out a meaningful answer. Please write in full sentences with correct spelling and punctuation. \
        If it makes sense use lists. If the context doen't contain the answer, \
        just respond that you are unable to find an answer. \
        ARTICLE: {article}
        QUESTION: {question}
        <end_of_turn>
        <start_of_turn>model\n
        ANSWER:"""
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(model=model, messages=messages)
    st.write("### Answer")
    st.write(response['message']['content'])
