import streamlit as st
import pdfplumber
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración profesional de la interfaz
st.set_page_config(page_title="AiVenair Expert", page_icon="🏢", layout="wide")

# Estilo personalizado para que se vea corporativo
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 AiVenair — Consultor Técnico de Ventas")
st.markdown("---")

# 1. Inicialización de estados de sesión
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

# Modelo de embeddings (Local y gratuito)
@st.cache_resource
def cargar_modelo_embeddings():
    # Este modelo es excelente para español y términos técnicos
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = cargar_modelo_embeddings()

# Barra lateral para gestión de archivos
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=100)
    st.header("Panel de Documentación")
    pdfs = st.file_uploader("Cargar Fichas Técnicas (PDF)", type="pdf", accept_multiple_files=True)
    
    if pdfs and st.button("🚀 Indexar Catálogo"):
        with st.spinner("Analizando especificaciones técnicas..."):
            texto_completo = ""
            for pdf in pdfs:
                with pdfplumber.open(pdf) as pdf_doc:
                    for page in pdf_doc.pages:
                        # pdfplumber extrae mejor el texto de tablas de ingeniería
                        texto_pag = page.extract_text()
                        if texto_pag:
                            texto_completo += f"\n--- DOCUMENTO: {pdf.name} ---\n{texto_pag}\n"
            
            # Dividimos en trozos grandes para no perder la relación de las tablas
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, 
                chunk_overlap=300,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(texto_completo)
            
            # Crear la base de datos vectorial local
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success(f"✅ {len(pdfs)} productos cargados correctamente.")

# Área de Chat Principal
if st.session_state.vector_store:
    # Contenedor para el historial de chat
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["texto"])

    # Entrada de usuario
    if pregunta := st.chat_input("Ej: ¿Qué mangueras de silicona manejamos para alta presión?"):
        with st.chat_message("user"):
            st.markdown(pregunta)
        
        # BÚSQUEDA AVANZADA: Recuperamos 15 fragmentos para tener visión global
        docs_relevantes = st.session_state.vector_store.similarity_search(pregunta, k=15)
        contexto_reducido = "\n\n".join([doc.page_content for doc in docs_relevantes])

        try:
            cliente = Groq(api_key=st.secrets["GROQ_KEY"])
            
            # El "Cerebro" del Asistente
            prompt_sistema = (
                "Eres el Asistente Experto de AiVenair, especializado en soluciones de mangueras y tuberías industriales. "
                "Tu objetivo es asistir al equipo de ventas y clientes con datos precisos.\n\n"
                "INSTRUCCIONES:\n"
                "1. Usa el siguiente CONTEXTO de fichas técnicas para responder:\n"
                f"--- CONTEXTO ---\n{contexto_reducido}\n----------------\n\n"
                "2. Si te preguntan por productos disponibles, haz una lista basada en los nombres de documentos y títulos en el contexto.\n"
                "3. Para datos técnicos (presión, temperatura, radio de curvatura), extrae los valores exactos.\n"
                "4. Si la información no está, invita al usuario a contactar a soporte técnico humano de Venair.\n"
                "5. Responde con un tono profesional, usando negritas para resaltar nombres de productos."
            )

            respuesta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": pregunta}
                ],
                temperature=0.2 # Bajo para evitar inventar datos técnicos
            )
            texto_respuesta = respuesta.choices[0].message.content

        except Exception as e:
            texto_respuesta = f"❌ Error en la consulta: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(texto_respuesta)
        
        # Guardar en historial
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": texto_respuesta})
else:
    st.info("👋 Bienvenido al sistema de soporte de AiVenair. Por favor, carga las fichas técnicas en el panel izquierdo para comenzar la consulta.")
