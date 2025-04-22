import streamlit as st
import os
from utils import load_translation, _
from home import home_page
from search_patient import search_patient_page
from predict import predict_page
from result import result_page
from preeclampsia_info import preeclampsia_info_page

# Initial state
if "language" not in st.session_state:
    st.session_state.language = "en"
if "page" not in st.session_state:
    st.session_state.page = "home"

# Load translation based on selected language
if "translations" not in st.session_state or st.session_state.language != st.session_state.get("last_language", ""):
    st.session_state.translations = load_translation(st.session_state.language)
    st.session_state.last_language = st.session_state.language

# Text direction and color
text_direction = "rtl" if st.session_state.language == "ar" else "ltr"
text_align = "right" if st.session_state.language == "ar" else "left"
text_color = "#003366"

# Interface styling
st.markdown(f"""
    <style>
        .stApp {{
            background-color: #EAF2FF;
            direction: {text_direction};
        }}
        h1, h2, h3, h4, h5, h6, p, label, .stTextInput, .stButton, .stRadio {{
            text-align: {text_align} !important;
            direction: {text_direction} !important;
            color: {text_color} !important;
        }}
        .stSidebar {{
            direction: {text_direction};
            text-align: {text_align};
            color: {text_color} !important;
        }}
        .stSidebar .stButton>button {{
            width: 220px;
            height: 48px;
            font-size: 16px;
            background-color: #e1eaf4;
            border: 1px solid #b0c4de;
            color: #003366;
            border-radius: 8px;
        }}
        .stSidebar .stButton>button:hover {{
            background-color: #d0e1f9;
            color: #001f33;
        }}
        .stSidebar .stButton>button:focus:not(:active) {{
            border: 2px solid #4a90e2;
            background-color: #ffffff;
        }}
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title(_("About our site"))
st.sidebar.write(_("We are Mama’s Hope, and we aspire to be a source of hope for every mother dreaming of a safe and stable pregnancy."))

if st.sidebar.button(_("Home page")):
    st.session_state.page = "home"
    st.rerun()

if st.sidebar.button(_("Start Prediction")):
    st.session_state.page = "predict"
    st.rerun()

if st.sidebar.button(_("Search for Patient")):
    st.session_state.page = "search_patient"
    st.rerun()

if st.sidebar.button(_("About Preeclampsia")):
    st.session_state.page = "preeclampsia_info"
    st.rerun()

# Language selection
language_choice = st.sidebar.radio(
    label=_("Change language"),
    options=["English", "Arabic"],
    index=0 if st.session_state.language == "en" else 1
)

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

# Page navigation
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
