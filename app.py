import streamlit as st
import pypdf
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuración de página
st.set_page_config(page_title="AiVenair RAG", page_icon="🏢", layout="wide")
st.title("🏢 AiVenair — Inteligencia Documental Ilimitada")

# 1. Inicializar estados
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

# Modelo de embeddings (Corre local en tu CPU, es GRATIS e ilimitado)
@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = cargar_modelo_embeddings()

with st.sidebar:
    st.header("📂 Base de Conocimiento")
    pdfs = st.file_uploader("Subir manuales o documentos de venta", type="pdf", accept_multiple_files=True)
    
    if pdfs and st.button("Indexar Documentos"):
        with st.spinner("Creando índice vectorial local..."):
            texto_completo = ""
            for pdf in pdfs:
                reader = pypdf.PdfReader(pdf)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: texto_completo += t + "\n"
            
            # Dividir el texto en trozos pequeños (Chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(texto_completo)
            
            # Crear base de datos vectorial en memoria (FAISS)
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success("✅ Documentos indexados. ¡Listo para vender!")

# Interfaz de Chat
if st.session_state.vector_store:
    # Mostrar historial
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])

    pregunta = st.chat_input("Consulta técnica o comercial...")
    
    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)
        
        # PASO CLAVE: Buscar solo los fragmentos relevantes (Top 3)
        docs_relevantes = st.session_state.vector_store.similarity_search(pregunta, k=3)
        contexto_reducido = "\n\n".join([doc.page_content for doc in docs_relevantes])

        try:
            cliente = Groq(api_key=st.secrets["GROQ_KEY"])
            
            # Enviamos POCO texto a Groq, ahorrando el 99% de tus tokens
            respuesta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"Eres el asistente de AiVenair. Responde basado EXCLUSIVAMENTE en este fragmento:\n\n{contexto_reducido}"},
                    {"role": "user", "content": pregunta}
                ],
                temperature=0.1 # Máxima precisión
            )
            texto_respuesta = respuesta.choices[0].message.content

        except Exception as e:
            texto_respuesta = f"❌ Error de API: {str(e)}"

        with st.chat_message("assistant"):
            st.write(texto_respuesta)
        
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": texto_respuesta})
else:
    st.info("👈 Por favor, sube tus documentos para activar el asistente.")
