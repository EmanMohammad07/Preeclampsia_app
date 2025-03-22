import streamlit as st
import os
from utils import load_translation, _
from home import home_page
from search_patient import search_patient_page
from predict import predict_page
from result import result_page
from preeclampsia_info import preeclampsia_info_page

# الحالة المبدئية
if "language" not in st.session_state:
    st.session_state.language = "en"
if "page" not in st.session_state:
    st.session_state.page = "home"

# تحميل الترجمة حسب اللغة المختارة
if "translations" not in st.session_state or st.session_state.language != st.session_state.get("last_language", ""):
    st.session_state.translations = load_translation(st.session_state.language)
    st.session_state.last_language = st.session_state.language

# اتجاه النص واللون
text_direction = "rtl" if st.session_state.language == "ar" else "ltr"
text_align = "right" if st.session_state.language == "ar" else "left"
text_color = "#003366"

# تنسيق الواجهة
st.markdown("""
    <style>
        .stApp {{
            background-color: #EAF2FF;
            direction: {direction};
        }}
        h1, h2, h3, h4, h5, h6, p, label, .stTextInput, .stButton, .stRadio {{
            text-align: {align} !important;
            direction: {direction} !important;
            color: {color} !important;
        }}
        .stSidebar {{
            direction: {direction};
            text-align: {align};
            color: {color} !important;
        }}
    </style>
""".format(direction=text_direction, align=text_align, color=text_color), unsafe_allow_html=True)

# الشريط الجانبي
st.sidebar.title(_("About our site"))
st.sidebar.write(_("We are Mama’s Hope, and we aspire to be a source of hope for every mother dreaming of a safe and stable pregnancy."))

if st.sidebar.button(_("Home page")):
    st.session_state.page = "home"
    st.rerun()

if st.sidebar.button(_("Search for Patient")):
    st.session_state.page = "search_patient"
    st.rerun()

if st.sidebar.button(_("Start Prediction")):
    st.session_state.page = "predict"
    st.rerun()

if st.sidebar.button(_("Preeclampsia Information")):
    st.session_state.page = "preeclampsia_info"
    st.rerun()

# اختيار اللغة
st.sidebar.markdown(f"{_('Change language')}")
language_choice = st.sidebar.radio(_("Select language"), ["English", "Arabic"], index=0 if st.session_state.language == "en" else 1)

if language_choice == "English" and st.session_state.language != "en":
    st.session_state.language = "en"
    st.session_state.translations = load_translation("en")
    st.session_state.last_language = "en"
    st.rerun()

elif language_choice == "Arabic" and st.session_state.language != "ar":
    st.session_state.language = "ar"
    st.session_state.translations = load_translation("ar")
    st.session_state.last_language = "ar"
    st.rerun()

# التنقل بين الصفحات
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "search_patient":
    search_patient_page()
elif st.session_state.page == "predict":
    predict_page()
elif st.session_state.page == "result":
    result_page()
elif st.session_state.page == "preeclampsia_info":
    preeclampsia_info_page()