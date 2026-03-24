import streamlit as st
import anthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

st.set_page_config(page_title="AiVenair", page_icon="🏢", layout="wide")
st.title("🏢 AiVenair — Asistente de Documentos")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.header("📂 Sube tus PDFs")
    pdfs = st.file_uploader("Selecciona archivos", type="pdf", accept_multiple_files=True)
    if pdfs and st.button("Procesar documentos"):
        with st.spinner("Analizando..."):
            docs = []
            for pdf in pdfs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(pdf.read())
                    loader = PyPDFLoader(f.name)
                    docs.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success(f"✅ {len(pdfs)} documento(s) listos")

if st.session_state.vectorstore:
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])

    pregunta = st.chat_input("Hazle una pregunta a tus documentos...")
    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)
        docs_relevantes = st.session_state.vectorstore.similarity_search(pregunta, k=4)
        contexto = "\n\n".join([d.page_content for d in docs_relevantes])
        cliente = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_KEY"])
        respuesta = cliente.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": f"Eres un asistente empresarial. Responde SOLO basándote en este contexto:\n\n{contexto}\n\nPregunta: {pregunta}\n\nSi no está en los documentos, dilo claramente."}]
        )
        texto = respuesta.content[0].text
        with st.chat_message("assistant"):
            st.write(texto)
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": texto})
else:
    st.info("👈 Sube tus PDFs en el panel izquierdo para comenzar")
