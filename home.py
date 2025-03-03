import streamlit as st
import os
from utils import _

def home_page():
    base_path = os.path.dirname(os.path.abspath(__file__)) # Define base path here
    # **تعديل هنا: استخدام use_column_width=True لجعل الصورة تستجيب لعرض العمود**
    st.image(os.path.join(base_path, "assets", "homepage.png"), use_container_width=True)

    st.markdown(f"""
    <h1>{_("Welcome to Mama's Hope")}</h1>
    <p>{_("In the world of medicine, where knowledge and professional expertise meet humanity, hope is born as the driving force that guides us toward a safer future for mothers and their babies.")}</p>
    <p>{_("At <b>Mama’s Hope</b>, we believe that every mother deserves the opportunity to live a healthy and secure life for herself and her child.")}</p>
    <p>{_("<b>Preeclampsia</b> is a serious medical condition that can significantly impact maternal and fetal health. However, early detection is key to improving outcomes.")}</p>
    <p>{_("Our website provides <b>healthcare professionals</b> with advanced tools and evidence-based insights to predict preeclampsia, empowering them to make timely and informed decisions.")}</p>
    <p>{_("We invite you to explore our website and be part of this meaningful journey. Every piece of information, every tool, and every step we take together brings us closer to a brighter and healthier future.")}</p>
    <p>{_("<b>Thank you for being part of Mama’s Hope.</b>")}</p>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    home_page() # This is for testing purposes
