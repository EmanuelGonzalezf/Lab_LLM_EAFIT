import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Configuración inicial de la página
st.set_page_config(
    page_title="LLM & NLP Exploration Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM & NLP Exploration Studio (Groq API)")
st.caption("Explora generación con LLMs, Tokenización, Bag of Words, Embeddings y Métricas de Similitud.")

# --- SIDEBAR: Configuración de API y Parámetros del Modelo ---
st.sidebar.header("🔑 Configuración de Groq API")
api_key = st.sidebar.text_input("Ingresa tu Groq API Key:", type="password", help="Obtén tu API key en https://console.groq.com/")

# Lista de modelos Groq actualizados (Llama 3, Mixtral, Gemma)
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

st.sidebar.header("⚙️ Parámetros del Modelo")
selected_model = st.sidebar.selectbox("Selecciona el Modelo LLM:", AVAILABLE_MODELS)
temperature = st.sidebar.slider("Temperatura:", min_value=0.0, max_value=2.0, value=0.7, step=0.1, help="Mayor temperatura = respuestas más creativas/aleatorias.")
max_tokens = st.sidebar.slider("Max Tokens de Salida:", min_value=64, max_value=4096, value=1024, step=64)
top_p = st.sidebar.slider("Top P (Nucleus Sampling):", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs([
    "🚀 Generación de Texto (Groq LLM)",
    "🔤 Tokenización & Bag of Words",
    "📐 Embeddings & Similitud Coseno"
])

# ==========================================
# PESTAÑA 1: GENERACIÓN DE TEXTO CON GROQ
# ==========================================
with tab1:
    st.header("Generación de Texto")
    
    system_prompt = st.text_area("System Prompt (Instrucción inicial):", value="Eres un asistente muy útil, preciso y conciso.", height=70)
    user_prompt = st.text_area("User Prompt (Mensaje del usuario):", value="Explica brevemente qué es la atención en Transformers.", height=100)

    if st.button("Generar Respuesta", type="primary"):
        if not api_key:
            st.error("⚠️ Por favor ingresa tu API Key de Groq en la barra lateral para continuar.")
        else:
            try:
                client = Groq(api_key=api_key)
                
                with st.spinner("Generando respuesta..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                
                response_text = chat_completion.choices[0].message.content
                usage = chat_completion.usage
                
                st.subheader("Respuesta del Modelo:")
                st.markdown(response_text)
                
                st.divider()
                st.markdown("### 📊 Métricas del Request & Tokens")
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", usage.prompt_tokens)
                col2.metric("Completion Tokens", usage.completion_tokens)
                col3.metric("Total Tokens", usage.total_tokens)
                
            except Exception as e:
                st.error(f"Error al conectar con la API de Groq: {e}")

# ==========================================
# PESTAÑA 2: TOKENIZACIÓN & BAG OF WORDS
# ==========================================
with tab2:
    st.header("Análisis de Tokens y Representación Bag of Words (BoW)")
    
    sample_text = st.text_area(
        "Ingresa texto para tokenizar e inspeccionar:",
        value="El procesamiento del lenguaje natural permite a los modelos comprender texto. Los modelos LLM analizan tokens.",
        height=100
    )
    
    if sample_text:
        col_tok, col_bow = st.columns(2)
        
        with col_tok:
            st.subheader("🔤 Tokenización Básica (Subpalabras / Palabras)")
            raw_tokens = sample_text.split()
            
            # Tabla de Tokens con ID numérico asignado para inspección
            tokens_df = pd.DataFrame({
                "Token ID": list(range(len(raw_tokens))),
                "Token": raw_tokens,
                "Longitud (caracteres)": [len(t) for t in raw_tokens]
            })
            
            st.dataframe(tokens_df, use_container_width=True)
            st.info(f"Total de palabras/tokens aproximados: **{len(raw_tokens)}**")

        with col_bow:
            st.subheader("🎒 Bag of Words (BoW Vector)")
            
            vectorizer = CountVectorizer()
            bow_matrix = vectorizer.fit_transform([sample_text])
            feature_names = vectorizer.get_feature_names_out()
            
            bow_df = pd.DataFrame(
                bow_matrix.toarray(),
                columns=feature_names,
                index=["Frecuencia"]
            ).T.sort_values(by="Frecuencia", ascending=False)
            
            st.dataframe(bow_df, use_container_width=True)

# ==========================================
# PESTAÑA 3: EMBEDDINGS & SIMILITUD COSENO
# ==========================================
with tab3:
    st.header("Métricas de Similitud y Vectores de Embeddings")
    st.write("Compara la semejanza semántica/vectorial entre dos textos utilizando representación **TF-IDF Vector Embeddings** y **Similitud Coseno**.")
    
    col_text1, col_text2 = st.columns(2)
    with col_text1:
        doc1 = st.text_area("Texto A:", value="Me gusta mucho la inteligencia artificial y el aprendizaje automático.", height=100)
    with col_text2:
        doc2 = st.text_area("Texto B:", value="La IA y el machine learning son campos fascinantes de la tecnología.", height=100)
        
    if doc1 and doc2:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([doc1, doc2])
        
        # Cálculo de Similitud Coseno
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        st.divider()
        st.subheader("📏 Métrica de Similitud")
        
        st.metric(
            label="Similitud Coseno (Cosine Similarity)",
            value=f"{cos_sim * 100:.2f}%",
            delta=f"{cos_sim:.4f} score"
        )
        
        st.progress(float(cos_sim))
        
        st.subheader("🔢 Vectores de Embeddings (TF-IDF Space)")
        vocabulary = tfidf.get_feature_names_out()
        
        embedding_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=vocabulary,
            index=["Vector Texto A", "Vector Texto B"]
        )
        
        st.dataframe(embedding_df.style.highlight_max(axis=0, color="#d1e7dd"), use_container_width=True)
