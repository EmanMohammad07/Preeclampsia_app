import streamlit as st
import os
import base64
from utils import _

def home_page():
    # تحديد المسار للصورة
    base_path = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_path, "assets", "homepage.png")

    # تنسيق CSS احترافي للألوان وتنسيق الصفحة
    st.markdown("""
    <style>
    .main-container {
        text-align: center;
        margin-top: 30px;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .main-title {
        font-size: 42px;
        font-weight: bold;
        background: -webkit-linear-gradient(90deg, #003366, #29648A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .subtitle, .highlight, .emphasis {
        font-size: 22px;
        color: #2A4D69;
        margin-bottom: 15px;
        text-align: center;
    }
    .highlight {
        font-size: 20px;
        color: #3B6E8F;
        font-weight: 600;
    }
    .emphasis {
        font-size: 18px;
        color: #5D697A;
        font-style: italic;
        margin-bottom: 30px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 40px;
    }
    .stButton>button {
        font-size: 10px;
        padding: 10px 30px;
        border-radius: 8px;
        background-color: #e1eaf4;
        color: #003366;
        border: 1px solid #b0c4de;
        width: 220px;
        height: 48px;
    }
    .stButton>button:hover {
        background-color: #d0e1f9;
        color: #001f33;
    }
    </style>
    """, unsafe_allow_html=True)

    # عرض الشعار في المنتصف
    st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{get_base64_image(logo_path)}' width='300'/></div>", unsafe_allow_html=True)

    # المحتوى النصي
    st.markdown(f"""
        <div class='main-container'>
            <h1 class='main-title'>{_('Welcome to Mama’s Hope')}</h1>
            <p class='subtitle'>{_("At Mama’s Hope, we empower healthcare professionals to predict preeclampsia early.")}</p>
            <p class='highlight'>{_("Together, we can build a safer, healthier future for mothers and babies.")}</p>
            <p class='emphasis'>{_("Your knowledge, our tools – one shared hope.")}</p>
        </div>
    """, unsafe_allow_html=True)

    # أزرار جنبًا إلى جنب
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(_("Start Prediction"), key="btn_predict"):
            st.session_state.page = "predict"
            st.rerun()

    with col2:
        if st.button("" + _("Search for Patient"), key="btn_view_records"):
            st.session_state.page = "search_patient"
            st.rerun()

    with col3:
        if st.button("" + _("About Preeclampsia"), key="btn_learn"):
            st.session_state.page = "preeclampsia_info"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# دالة لتحويل الصورة إلى base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == '__main__':
    home_page()