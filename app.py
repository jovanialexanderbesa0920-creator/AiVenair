import streamlit as st
import anthropic
import pypdf

st.set_page_config(page_title="AiVenair", page_icon="🏢", layout="wide")
st.title("🏢 AiVenair — Asistente de Documentos")

if "texto_pdfs" not in st.session_state:
    st.session_state.texto_pdfs = ""
if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.header("📂 Sube tus PDFs")
    pdfs = st.file_uploader("Selecciona archivos", type="pdf", accept_multiple_files=True)
    if pdfs and st.button("Procesar documentos"):
        with st.spinner("Analizando..."):
            texto_total = ""
            for pdf in pdfs:
                reader = pypdf.PdfReader(pdf)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        texto_total += t + "\n"
            st.session_state.texto_pdfs = texto_total[:15000]
        st.success(f"✅ {len(pdfs)} documento(s) listos")

if st.session_state.texto_pdfs:
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])
    pregunta = st.chat_input("Hazle una pregunta a tus documentos...")
    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)
        cliente = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_KEY"])
        respuesta = cliente.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"Eres un asistente empresarial. Responde SOLO basándote en este contexto:\n\n{st.session_state.texto_pdfs}\n\nPregunta: {pregunta}\n\nSi no está en los documentos, dilo claramente."
            }]
        )
        texto = respuesta.content[0].text
        with st.chat_message("assistant"):
            st.write(texto)
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": texto})
else:
    st.info("👈 Sube tus PDFs en el panel izquierdo para comenzar")
