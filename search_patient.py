import streamlit as st
from utils import _, load_patient_data, search_patient_by_id

def search_patient_page():
    st.title("🔍 " + _("Search for Patient"))

    patient_id = st.text_input(_("Enter Patient ID"))

    if st.button(_("Search")):
        if not patient_id:
            st.warning(_("Please enter a Patient ID to search."))
            return

        # Check if the patient ID is exactly 6 digits
        if not (patient_id.isdigit() and len(patient_id) == 6):
            st.warning(_("Patient ID must be exactly 6 digits."))
            return

        patient_data = load_patient_data()

        if patient_data is not None:
            # Search for patient by ID
            patient_record = search_patient_by_id(patient_id, patient_data)

            if patient_record is not None and not patient_record.empty:
                # Sort by patient_id just in case multiple records exist
                patient_record_sorted = patient_record.sort_values(by='patient_id')

                st.success(_("Patient found successfully!"))
                st.subheader(_("Patient Details"))
                st.dataframe(patient_record_sorted, use_container_width=True)
            else:
                st.warning(_("No patient found with this ID."))
        else:
            st.error(_("Failed to load patient data."))

    st.markdown("---")
    if st.button(_("Back to Home")):
        st.session_state.page = "home"
        st.rerun()

# Run the page function
if __name__ == "__main__":
    search_patient_page()