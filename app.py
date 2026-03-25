import streamlit as st
import pdfplumber
import re
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="AiVenair Expert v3", page_icon="🏢", layout="wide")

# --- ESTADOS DE SESIÓN ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = cargar_modelo_embeddings()

# --- FUNCIONES DE PROCESAMIENTO ---
def extraer_datos_venair(file):
    texto_con_contexto = ""
    nombre_producto = "Producto Desconocido"
    
    with pdfplumber.open(file) as pdf:
        # 1. Intentar capturar el nombre del producto en la primera página
        primera_página = pdf.pages[0].extract_text()
        match = re.search(r"Vena®\s+[\w\d\s]+", primera_página)
        if match:
            nombre_producto = match.group(0).strip()
        
        # 2. Extraer texto y tablas de cada página
        for i, page in enumerate(pdf.pages):
            # Extraer tablas formateadas para mantener la estructura de la ficha técnica
            tabla = page.extract_table()
            texto_tabla = ""
            if tabla:
                for fila in tabla:
                    texto_tabla += " | ".join([str(celda) for celda in fila if celda]) + "\n"
            
            # Combinar texto normal y tabla, inyectando el nombre del producto siempre
            contenido = page.extract_text() or ""
            texto_con_contexto += f"\n[PRODUCTO: {nombre_producto}] [PÁGINA: {i+1}]\n{contenido}\n{texto_tabla}\n"
            
    return texto_con_contexto

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🏢 Catálogo AiVenair")
    archivos = st.file_uploader("Subir fichas técnicas (PDF)", type="pdf", accept_multiple_files=True)
    
    if archivos and st.button("Actualizar Base de Datos"):
        with st.spinner("Procesando catálogos y tablas técnicas..."):
            texto_total = ""
            for arc in archivos:
                texto_total += extraer_datos_venair(arc)
            
            # Chunks más grandes para no romper las tablas de la página 2
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400
            )
            docs = text_splitter.split_text(texto_total)
            st.session_state.vector_store = FAISS.from_texts(docs, embeddings)
            st.success(f"✅ {len(archivos)} productos listos.")

# --- CHAT ---
if st.session_state.vector_store:
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])

    if pregunta := st.chat_input("¿Qué presión de trabajo tiene la SIL 640 de 1 pulgada?"):
        with st.chat_message("user"):
            st.write(pregunta)

        # Buscamos en 15 fragmentos para cubrir múltiples productos
        docs = st.session_state.vector_store.similarity_search(pregunta, k=15)
        contexto = "\n\n".join([d.page_content for d in docs])

        try:
            cliente = Groq(api_key=st.secrets["GROQ_KEY"])
            
            prompt_sistema = (
                "Eres el experto técnico de AiVenair. Responde consultas basadas en las fichas técnicas proporcionadas.\n"
                "REGLAS:\n"
                "1. Si te preguntan por productos, revisa las etiquetas [PRODUCTO: ...] en el contexto.\n"
                "2. Para datos de presión o diámetro, busca en las filas de las tablas (ej: '1 | 25 | 6.7 Bar').\n"
                "3. La SIL 640 tiene un rango de -60°C a +180°C. Si preguntan por temperatura, usa esos valores.\n"
                f"\nCONTEXTO TÉCNICO:\n{contexto}"
            )

            res = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": pregunta}],
                temperature=0.1
            )
            respuesta_ia = res.choices[0].message.content
        except Exception as e:
            respuesta_ia = f"❌ Error: {str(e)}"

        with st.chat_message("assistant"):
            st.write(respuesta_ia)
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": respuesta_ia})
else:
    st.info("Por favor, carga las fichas técnicas para comenzar.")
