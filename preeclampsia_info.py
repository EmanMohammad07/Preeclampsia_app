import streamlit as st
from utils import _

def preeclampsia_info_page():
    # Adding the preeclampsia information page content
    st.title(_("Preeclampsia"))
    
    st.header(_("What is Preeclampsia?"))
    st.write(_("Preeclampsia is a serious medical condition that occurs during pregnancy, characterized by high blood pressure and the presence of protein in the urine. If left untreated, it can lead to severe complications for both the mother and baby."))

    st.header(_("Symptoms of Preeclampsia"))
    st.write(_("The symptoms of preeclampsia can range from mild to severe and may include:"))
    st.write(f"""
    - {_("High blood pressure (above 140/90 mmHg)")}
    - {_("Unusual swelling in the hands and feet")}
    - {_("Severe and sudden headaches")}
    - {_("Visual disturbances such as blurred vision or seeing light spots")}
    - {_("Pain in the upper abdomen")}
    - {_("Continuous nausea or vomiting")}
    """)

    st.header(_("Risk Factors"))
    st.write(_("Some of the risk factors for preeclampsia include:"))
    st.write(f"""
    - {_("First pregnancy")}
    - {_("Family history of preeclampsia")}
    - {_("Carrying twins or more")}
    - {_("Obesity")}
    - {_("Chronic high blood pressure")}
    """)

    # Adding a back button to go to the home page
    if st.button(_("Back to Home Page")):
        st.session_state.page = "home"
        st.rerun()