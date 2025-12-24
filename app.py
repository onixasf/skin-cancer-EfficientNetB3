import streamlit as st
import requests
import numpy as np
import plotly.express as px
from PIL import Image
import io

# =========================================================
# CONFIG
# =========================================================
API_URL = "https://onixasf-skin-cancer-efficientnetb3.hf.space/run/predict"

CLASS_DETAILS = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

st.set_page_config(
    page_title="Skin Cancer Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <h1 style='text-align:center;'>🩺 Skin Cancer Classification Dashboard</h1>
    <p style='text-align:center; color:gray;'>
    EfficientNetB3 · HAM10000 Dataset · Final Project Sains Data
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📌 Overview",
    "📈 Model Explanation",
    "🧪 Try Prediction"
])

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.header("📌 Dataset & Project Overview")

    st.write("""
    Proyek ini mengembangkan sistem **klasifikasi kanker kulit**
    menggunakan **dataset HAM10000** yang terdiri dari 7 kelas lesi kulit.

    Model dilatih menggunakan **EfficientNetB3** dengan pendekatan
    *transfer learning* dan *fine-tuning* untuk meningkatkan akurasi klasifikasi.
    """)

    st.subheader("🧬 Diagnosis Classes")
    for k, v in CLASS_DETAILS.items():
        st.markdown(f"- **{k.upper()}** : {v}")

    st.info("⚠️ Sistem ini bersifat edukatif dan **bukan pengganti diagnosis medis**.")

# =========================================================
# TAB 2 – MODEL EXPLANATION
# =========================================================
with tab2:
    st.header("📈 Model Explanation")

    st.write("""
    **EfficientNetB3** merupakan arsitektur CNN modern yang
    menyeimbangkan **depth, width, dan resolution** secara optimal.

    **Konfigurasi model:**
    - Input size: 300 × 300
    - Optimizer: Adam
    - Loss: Categorical Crossentropy
    - Dataset balancing
    """)

    st.success("""
    Model menunjukkan performa stabil dan peningkatan signifikan
    dibanding CNN baseline, terutama pada kelas mayoritas dan minoritas.
    """)

# =========================================================
# TAB 3 – PREDICTION (API)
# =========================================================
with tab3:
    st.header("🧪 Skin Lesion Prediction (API Based)")

    uploaded_file = st.file_uploader(
        "Upload gambar lesi kulit (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        with st.spinner("🔍 Mengirim gambar ke model..."):
            image_bytes = uploaded_file.read()

            response = requests.post(
                API_URL,
                files={"file": image_bytes},
                timeout=60
            )

        if response.status_code == 200:
            result = response.json()

            pred = result["predicted_class"]
            conf = result["confidence"]
            probs = result["probabilities"]

            st.success(
                f"### 🧠 Prediksi: **{pred.upper()}** — {CLASS_DETAILS[pred]}"
            )
            st.write(f"**Confidence:** {conf:.4f}")

            fig = px.bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                labels={"x": "Class", "y": "Probability"},
                title="Probability Distribution",
                color=list(probs.values()),
                color_continuous_scale="Blues"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ Gagal mendapatkan prediksi dari API")

# =========================================================
# FOOTER
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
**🩺 Skin Cancer Classification Dashboard**  

👩🏻‍💻 **Onixa Shafa Putri Wibowo**  
📘 Final Project – Sains Data
""")
