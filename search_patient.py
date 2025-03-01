import streamlit as st
from utils import _, load_patient_data, search_patient_by_id

def search_patient_page():
    st.title(_("Search for Patient"))
    patient_id = st.text_input(_("Enter Patient ID"))

    if st.button(_("Search")):
        # تحميل البيانات من الملف
        patient_data = load_patient_data()

        if patient_data is not None:
            # البحث عن المريض بواسطة Patient ID
            patient_record = search_patient_by_id(patient_id, patient_data)

            if patient_record is not None:
                st.write(patient_record)  # عرض البيانات الخاصة بالمريض
            else:
                st.warning(_("No patient found with this ID."))
        else:
            st.error(_("Failed to load patient data."))

    if st.button(_("Back to Home")):
        st.session_state.page = "home"
        st.rerun()

if __name__ == '__main__':
    search_patient_page() # For testing