import streamlit as st
import os
from utils import _
import base64

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
        }
        .main-title {
            font-size: 42px;
            font-weight: bold;
            background: -webkit-linear-gradient(90deg, #003366, #29648A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 22px;
            color: #2A4D69;
            margin-bottom: 15px;
        }
        .highlight {
            font-size: 20px;
            color: #3B6E8F;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .emphasis {
            font-size: 18px;
            color: #5D697A;
            font-style: italic;
            margin-bottom: 30px;
        }
        .button-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            margin-top: 40px;
        }
        .stButton>button {
            font-size: 16px;
            padding: 10px 30px;
            border-radius: 8px;
            background-color: #e1eaf4;
            color: #003366;
            border: 1px solid #b0c4de;
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

    # أزرار تحت بعض
    st.markdown("<div class='button-column'>", unsafe_allow_html=True)

    if st.button(_("Start Prediction"), key="btn_predict"):
        st.session_state.page = "predict"
        st.rerun()

    if st.button("" + _("View Patient Records"), key="btn_view_records"):
        st.session_state.page = "search_patient"
        st.rerun()

    if st.button("" + _("Learn About Preeclampsia"), key="btn_learn"):
        st.session_state.page = "preeclampsia_info"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# دالة لتحويل الصورة إلى base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == '__main__':
    home_page()