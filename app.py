import streamlit as st
import pdfplumber
import re
import os
from groq import Groq
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración de interfaz
st.set_page_config(page_title="AiVenair - Soporte Ininterrumpido", page_icon="🏢", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

@st.cache_resource
def cargar_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = cargar_modelo_embeddings()

# --- PROCESAMIENTO DE FICHAS TÉCNICAS ---
def procesar_pdf_venair(file):
    texto_estructurado = ""
    with pdfplumber.open(file) as pdf:
        # [span_0](start_span)[span_1](start_span)Detectar nombre del producto (ej: Vena® SIL 640)[span_0](end_span)[span_1](end_span)
        primera_pag = pdf.pages[0].extract_text() or ""
        nombre_prod = "Producto Venair"
        match = re.search(r"Vena®\s+[\w\d\s]+", primera_pag)
        if match:
            nombre_prod = match.group(0).strip()
            
        for i, page in enumerate(pdf.pages):
            contenido = page.extract_text() or ""
            # [span_2](start_span)Extraer tablas para datos de presión y diámetros[span_2](end_span)
            tabla = page.extract_table()
            texto_tabla = ""
            if tabla:
                for fila in tabla:
                    texto_tabla += " | ".join([str(c) for c in fila if c]) + "\n"
            
            texto_estructurado += f"\n[PRODUCTO: {nombre_prod}] [PÁGINA: {i+1}]\n{contenido}\n{texto_tabla}\n"
    return texto_estructurado

# --- LÓGICA DE INTELIGENCIA (DUAL API) ---
def consultar_ia(contexto, pregunta):
    prompt_sistema = (
        "Eres el experto técnico de AiVenair. Responde usando SOLO el contexto de fichas técnicas.\n"
        "Si preguntan por productos, enuméralos según las etiquetas [PRODUCTO: ...].\n"
        "Si preguntan por especificaciones (presión, temperatura, radio), usa las tablas del contexto.\n"
        f"CONTEXTO:\n{contexto}"
    )

    # INTENTO 1: GROQ (Llama 3.3)
    try:
        cliente_groq = Groq(api_key=st.secrets["GROQ_KEY"])
        res = cliente_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": pregunta}],
            temperature=0.1
        )
        return res.choices[0].message.content, "Groq (Principal)"
    
    except Exception as e:
        # INTENTO 2: RESPALDO GOOGLE GEMINI (Si Groq falla o agota cuota)
        try:
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            # En Gemini el prompt de sistema se concatena o se usa system_instruction
            full_prompt = f"{prompt_sistema}\n\nUsuario pregunta: {pregunta}"
            res_gemini = model.generate_content(full_prompt)
            return res_gemini.text, "Gemini (Respaldo)"
        except Exception as e_gen:
            return f"Error crítico en ambas APIs: {str(e_gen)}", "Ninguno"

# --- INTERFAZ ---
with st.sidebar:
    st.header("🏢 Administración de Fichas")
    archivos = st.file_uploader("Subir PDFs de Venair", type="pdf", accept_multiple_files=True)
    
    if archivos and st.button("Indexar Catálogo"):
        with st.spinner("Sincronizando base de datos técnica..."):
            base_datos_texto = ""
            for arc in archivos:
                base_datos_texto += procesar_pdf_venair(arc)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            fragmentos = splitter.split_text(base_datos_texto)
            st.session_state.vector_store = FAISS.from_texts(fragmentos, embeddings)
            st.success("✅ Catálogo actualizado.")

if st.session_state.vector_store:
    for m in st.session_state.historial:
        with st.chat_message(m["rol"]): st.write(m["texto"])

    if pregunta := st.chat_input("¿Qué presión resiste la SIL 640 de 1/2 pulgada?"):
        with st.chat_message("user"): st.write(pregunta)

        # RAG: Buscar 15 fragmentos más relevantes
        docs = st.session_state.vector_store.similarity_search(pregunta, k=15)
        contexto_rag = "\n\n".join([d.page_content for d in docs])

        respuesta, motor = consultar_ia(contexto_rag, pregunta)
        
        with st.chat_message("assistant"):
            st.markdown(respuesta)
            st.caption(f"Respondido por motor: {motor}")
        
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": respuesta})
else:
    st.info("👈 Cargue las fichas técnicas para activar el soporte.")
