import streamlit as st
import pdfplumber
import re
from groq import Groq
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración de página
st.set_page_config(page_title="AiVenair Expert v5", page_icon="🏢", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

@st.cache_resource
def cargar_modelos():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = cargar_modelos()

# --- EXTRACCIÓN MEJORADA DE PRODUCTOS ---
def procesar_pdf_tecnico(file):
    texto_full = ""
    with pdfplumber.open(file) as pdf:
        # Extraer nombre del producto (Vena® ...)
        primera_página = pdf.pages[0].extract_text() or ""
        match = re.search(r"Vena®\s+[A-Z0-9\-\s]+", primera_página, re.IGNORECASE)
        producto_id = match.group(0).strip() if match else file.name
        
        for i, page in enumerate(pdf.pages):
            # Extraer tablas con formato markdown para que la IA las entienda mejor
            tablas = page.extract_tables()
            txt_tablas = ""
            if tablas:
                for tabla in tablas:
                    for fila in tabla:
                        fila_limpia = [str(c).replace("\n", " ") for c in fila if c]
                        txt_tablas += " | ".join(fila_limpia) + "\n"
            
            contenido = page.extract_text() or ""
            texto_full += f"\n--- PRODUCTO: {producto_id} | PÁGINA: {i+1} ---\n{contenido}\n{txt_tablas}\n"
    return texto_full

# --- CEREBRO CON CORRECCIÓN DE ERROR 404 ---
def consultar_ia(contexto, pregunta):
    sistema = (
        "Eres el Ingeniero de Aplicaciones de AiVenair. Tu objetivo es dar datos exactos.\n"
        "REGLAS:\n"
        "1. Identifica el producto por la etiqueta PRODUCTO: ...\n"
        "2. Usa las tablas para dar valores de Presión de Trabajo, Rotura y Radio de Curvatura.\n"
        "3. Si la información no está, indica que se debe consultar al departamento técnico.\n"
        f"CONTEXTO TÉCNICO:\n{contexto}"
    )

    # 1. INTENTO CON GROQ
    try:
        cliente_groq = Groq(api_key=st.secrets["GROQ_KEY"])
        res = cliente_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": sistema}, {"role": "user", "content": pregunta}],
            temperature=0.1
        )
        return res.choices[0].message.content, "Groq (Llama 3.3)"
    
    except Exception:
        # 2. RESPALDO CON GEMINI (Corrección de error 404)
        try:
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            # Usamos el nombre del modelo sin versiones beta para mayor estabilidad
            model = genai.GenerativeModel('gemini-1.5-flash')
            # En Gemini pasamos el sistema como parte del contenido inicial
            response = model.generate_content(f"{sistema}\n\nPregunta: {pregunta}")
            return response.text, "Google Gemini (Flash)"
        except Exception as e:
            return f"Error técnico: Ambas APIs están fuera de servicio. {str(e)}", "Falla"

# --- INTERFAZ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
    st.title("Panel Técnico")
    archivos = st.file_uploader("Subir fichas técnicas", type="pdf", accept_multiple_files=True)
    
    if archivos and st.button("🚀 Procesar Catálogo"):
        with st.spinner("Indexando especificaciones..."):
            texto_total = ""
            for arc in archivos:
                texto_total += procesar_pdf_tecnico(arc)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            chunks = splitter.split_text(texto_total)
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success(f"{len(archivos)} productos cargados.")

# Área de conversación
if st.session_state.vector_store:
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]): st.write(msg["texto"])

    if q := st.chat_input("¿Qué presión resiste la Technipur-VAC de 50mm?"):
        with st.chat_message("user"): st.write(q)
        
        # RAG: Buscamos en los 10 fragmentos más relevantes
        docs = st.session_state.vector_store.similarity_search(q, k=10)
        ctx = "\n\n".join([d.page_content for d in docs])
        
        resp, motor = consultar_ia(ctx, q)
        
        with st.chat_message("assistant"):
            st.markdown(resp)
            st.caption(f"Motor: {motor}")
        
        st.session_state.historial.append({"rol": "user", "texto": q})
        st.session_state.historial.append({"rol": "assistant", "texto": resp})
else:
    st.info("Carga los PDFs en el panel izquierdo para empezar.")
