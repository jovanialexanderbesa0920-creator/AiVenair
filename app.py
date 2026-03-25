import streamlit as st
import pdfplumber
import re
import os
from groq import Groq
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuración de la Interfaz Profesional
st.set_page_config(page_title="AiVenair Expert v4", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 AiVenair — Consultor Técnico Inteligente")
st.caption("Respaldo Dual: Llama 3.3 (Groq) + Gemini 1.5 Flash (Google)")

# 2. Inicialización de Memoria
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "historial" not in st.session_state:
    st.session_state.historial = []

@st.cache_resource
def cargar_embeddings():
    # Modelo optimizado para español técnico
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings_model = cargar_embeddings()

# 3. Procesamiento Avanzado de Fichas Técnicas
def procesar_ficha_venair(file):
    texto_final = ""
    with pdfplumber.open(file) as pdf:
        # Extraer el nombre del producto del encabezado (Ej: Vena® SIL 640)
        texto_inicio = pdf.pages[0].extract_text() or ""
        # [span_0](start_span)[span_1](start_span)Buscamos el patrón "Vena®" seguido de su modelo[span_0](end_span)[span_1](end_span)
        match = re.search(r"Vena®\s+[\w\d\s]+", texto_inicio)
        id_producto = match.group(0).strip() if match else "Producto Venair"
        
        for num, pagina in enumerate(pdf.pages):
            # Texto plano
            cuerpo = pagina.extract_text() or ""
            
            # [span_2](start_span)Tablas (Vital para presiones y diámetros en pág 2)[span_2](end_span)
            tabla_datos = pagina.extract_table()
            txt_tabla = ""
            if tabla_datos:
                for fila in tabla_datos:
                    # Limpiamos valores nulos y unimos con separadores
                    fila_limpia = [str(item).replace('\n', ' ') for item in fila if item]
                    txt_tabla += " | ".join(fila_limpia) + "\n"
            
            # Inyectamos el nombre del producto en cada página para que la IA no pierda el contexto
            texto_final += f"\n[DOCUMENTO: {id_producto}] [PAG: {num+1}]\n{cuerpo}\n{txt_tabla}\n"
            
    return texto_final

# 4. Cerebro con Redundancia (Dual API)
def generar_respuesta(contexto, prompt_usuario):
    instrucciones = (
        "Eres el Ingeniero de Soporte Técnico de AiVenair. Tu misión es dar datos precisos.\n"
        "REGLAS:\n"
        "1. Usa el contexto para responder. Si mencionan presiones, busca en las tablas extraídas.\n"
        "2. Identifica el producto por la etiqueta [DOCUMENTO: ...].\n"
        "3. [span_3](start_span)La Vena® SIL 640 soporta de -60°C a +180°C[span_3](end_span). Si preguntan por calor extremo, menciónalo.\n"
        f"\nCONTEXTO TÉCNICO:\n{contexto}"
    )

    # INTENTO A: GROQ
    try:
        client = Groq(api_key=st.secrets["GROQ_KEY"])
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": instrucciones},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.1
        )
        return chat_completion.choices[0].message.content, "Motor Principal (Groq)"
    
    except Exception as e:
        # INTENTO B: GEMINI (Respaldo)
        try:
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            # Usamos el nombre del modelo sin prefijos extraños para evitar el 404
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_query = f"{instrucciones}\n\nPregunta del cliente: {prompt_usuario}"
            response = model.generate_content(full_query)
            return response.text, "Motor de Respaldo (Gemini)"
        except Exception as e_gem:
            return f"Lo siento, ambos sistemas de IA están saturados. Error: {str(e_gem)}", "Falla Total"

# 5. Interfaz de Usuario
with st.sidebar:
    st.header("⚙️ Configuración")
    pdf_docs = st.file_uploader("Cargar Catálogos PDF", type="pdf", accept_multiple_files=True)
    
    if pdf_docs and st.button("Indexar Productos"):
        with st.spinner("Analizando materiales y tablas técnicas..."):
            texto_acumulado = ""
            for doc in pdf_docs:
                texto_acumulado += procesar_ficha_venair(doc)
            
            # Dividir en fragmentos respetando el tamaño de las tablas
            splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300)
            chunks = splitter.split_text(texto_acumulado)
            
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings_model)
            st.success(f"✅ {len(pdf_docs)} productos indexados correctamente.")

# Mostrar chat
if st.session_state.vector_store:
    for chat in st.session_state.historial:
        with st.chat_message(chat["rol"]):
            st.write(chat["texto"])

    if user_input := st.chat_input("¿Cuál es el radio de curvatura o presión de la SIL 640?"):
        with st.chat_message("user"):
            st.write(user_input)
        
        # RAG: Recuperar los 12 fragmentos más parecidos
        busqueda = st.session_state.vector_store.similarity_search(user_input, k=12)
        contexto_rag = "\n\n".join([res.page_content for res in busqueda])
        
        respuesta_final, motor_usado = generar_respuesta(contexto_rag, user_input)
        
        with st.chat_message("assistant"):
            st.markdown(respuesta_final)
            st.caption(f"Fuente: {motor_usado}")
        
        st.session_state.historial.append({"rol": "user", "texto": user_input})
        st.session_state.historial.append({"rol": "assistant", "texto": respuesta_final})
else:
    st.info("👋 Por favor, sube las fichas técnicas en el panel lateral para comenzar.")
