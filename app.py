import streamlit as st
import pypdf
import requests
import json

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
            st.session_state.texto_pdfs = texto_total[:12000]
        st.success(f"✅ {len(pdfs)} documento(s) listos")

if st.session_state.texto_pdfs:
    for msg in st.session_state.historial:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])

    pregunta = st.chat_input("Hazle una pregunta a tus documentos...")
    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        try:
            api_key = st.secrets["GEMINI_KEY"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Eres un asistente empresarial. Responde SOLO basándote en este contexto:\n\n{st.session_state.texto_pdfs}\n\nPregunta: {pregunta}\n\nSi no está en los documentos, dilo claramente."
                    }]
                }]
            }
            response = requests.post(url, json=payload, timeout=30)
            data = response.json()

            if "candidates" in data:
                texto = data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                texto = f"❌ Error Gemini: {data['error']['message']}"
            else:
                texto = f"❌ Respuesta inesperada: {json.dumps(data)}"

        except requests.exceptions.Timeout:
            texto = "❌ Timeout: Gemini tardó demasiado, intenta de nuevo."
        except KeyError as e:
            texto = f"❌ KeyError: {str(e)} — Respuesta: {json.dumps(data)}"
        except Exception as e:
            texto = f"❌ Error inesperado: {str(e)}"

        with st.chat_message("assistant"):
            st.write(texto)
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append({"rol": "assistant", "texto": texto})

else:
    st.info("👈 Sube tus PDFs en el panel izquierdo para comenzar")
