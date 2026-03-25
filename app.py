import streamlit as st
import pypdf
from groq import Groq
import re

st.set_page_config(page_title="AiVenair", page_icon="🏢", layout="wide")
st.title("🏢 AiVenair — Asistente Corporativo Especializado")

# 1. Inicialización de estado dividido: Uno visual, otro estructurado para la API
if "texto_pdfs" not in st.session_state:
    st.session_state.texto_pdfs = ""
if "historial_interfaz" not in st.session_state:
    st.session_state.historial_interfaz = []
if "historial_api" not in st.session_state:
    st.session_state.historial_api = []

def limpiar_texto(texto):
    """Elimina saltos de línea y espacios excesivos para optimizar el consumo de tokens."""
    texto = re.sub(r'\n+', '\n', texto)
    return texto.strip()

with st.sidebar:
    st.header("📂 Gestión Documental")
    pdfs = st.file_uploader("Seleccione los manuales técnicos o comerciales (PDF)", type="pdf", accept_multiple_files=True)
    
    if pdfs and st.button("Procesar e Indexar Documentos"):
        with st.spinner("Extrayendo y estructurando la base de conocimiento..."):
            texto_total = ""
            for pdf in pdfs:
                try:
                    reader = pypdf.PdfReader(pdf)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            texto_total += t + "\n"
                except Exception as e:
                    st.error(f"Error al procesar el archivo {pdf.name}: {e}")
            
            # Se elimina el truncamiento arbitrario. 
            # Llama 3.3 70B tiene capacidad sobrada para procesar manuales completos de una sola vez.
            st.session_state.texto_pdfs = limpiar_texto(texto_total)
            
        st.success(f"✅ Análisis completado: {len(pdfs)} documento(s) listos para consulta operativa.")

# Flujo Principal de Interacción
if st.session_state.texto_pdfs:
    # Renderizar el historial en la interfaz de usuario
    for msg in st.session_state.historial_interfaz:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])

    pregunta = st.chat_input("Ingrese la consulta técnica o de ventas requerida...")
    
    if pregunta:
        # Mostrar la consulta en pantalla
        with st.chat_message("user"):
            st.write(pregunta)

        # Configuración del marco de actuación (System Prompt)
        instruccion_sistema = (
            "Actúa como el asistente corporativo y de ingeniería de ventas de AiVenair. "
            "Tu única fuente de verdad y conocimiento es el siguiente contexto documental:\n\n"
            f"<contexto>\n{st.session_state.texto_pdfs}\n</contexto>\n\n"
            "Reglas estrictas de operación:\n"
            "1. Prioriza la precisión técnica y los datos comerciales exactos.\n"
            "2. Si un dato no figura explícitamente en el texto, indica con profesionalismo que la información requiere escalamiento a un ingeniero especializado. No inventes especificaciones.\n"
            "3. Mantén un tono sumamente formal, persuasivo y estructurado."
        )

        # Construcción de la carga útil (Payload) con memoria conversacional
        mensajes_api = [{"role": "system", "content": instruccion_sistema}]
        mensajes_api.extend(st.session_state.historial_api)
        mensajes_api.append({"role": "user", "content": pregunta})

        try:
            cliente = Groq(api_key=st.secrets["GROQ_KEY"])
            
            respuesta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=mensajes_api,
                max_tokens=2048,
                temperature=0.1, # Se reduce a 0.1 para forzar respuestas factuales y mitigar alucinaciones
                top_p=0.9
            )
            texto = respuesta.choices[0].message.content

        except Exception as e:
            texto = f"❌ Error de procesamiento del modelo: {str(e)}"

        # Mostrar respuesta en pantalla
        with st.chat_message("assistant"):
            st.write(texto)
            
        # Actualización de memoria visual (Streamlit)
        st.session_state.historial_interfaz.append({"rol": "user", "texto": pregunta})
        st.session_state.historial_interfaz.append({"rol": "assistant", "texto": texto})
        
        # Actualización de memoria estructurada (Groq API)
        st.session_state.historial_api.append({"role": "user", "content": pregunta})
        st.session_state.historial_api.append({"role": "assistant", "content": texto})

else:
    st.info("👈 Por favor, inicialice el sistema cargando los documentos requeridos en el panel de control izquierdo.")
