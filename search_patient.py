import streamlit as st
from utils import _, load_patient_data, search_patient_by_id

def search_patient_page():
    st.title("🔍 " + _("Search for Patient"))

    patient_id = st.text_input(_("Enter Patient ID"))

    if st.button(_("Search")):
        if not patient_id:
            st.warning(_("Please enter a Patient ID to search."))
            return

        patient_data = load_patient_data()

        if patient_data is not None:
            patient_record = search_patient_by_id(patient_id, patient_data)

            if patient_record is not None:
                st.success(_("Patient found successfully!"))
                st.subheader(_("Patient Details"))
                st.dataframe(patient_record, use_container_width=True)  # عرض البيانات بشكل جدول عمودي
            else:
                st.warning(_("No patient found with this ID."))
        else:
            st.error(_("Failed to load patient data."))

    st.markdown("---")
    if st.button(_("Back to Home")):
        st.session_state.page = "home"
        st.rerun()

    if __name__ == "__main__":
       search_patient_page()