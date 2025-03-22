import streamlit as st
import pandas as pd
from utils import _

def result_page():
    st.title("🩺 " + _("Prediction Result"))

    result = st.session_state.get("result", None)
    derived_table = st.session_state.get("derived_table", None)
    patient_info = st.session_state.get("patient_info", {})

    if result is None:
        st.error(_("No prediction result available. Please go back and run the prediction first."))
        if st.button(_("🔙 Back to Prediction")):
            st.session_state.page = "predict"
            st.rerun()
        return

    # عرض النتيجة والتوصية
    if "no preeclampsia" in result.lower():
        st.success(f" {result}")
        recommendation = _("No immediate medical concerns. Continue monitoring health regularly.")
        st.info(f" {recommendation}")
    else:
        st.error(f"⚠️ {result}")
        recommendation = _("We recommend immediate medical attention.")
        st.warning(f" {recommendation}")
        st.markdown("💡 *" + _("This condition requires close medical supervision. Please consult your physician as soon as possible.") + "*")

    # عرض القيم المشتقة
    if derived_table is not None:
        st.markdown("## " + _("Derived Clinical Features"))
        styled_table = derived_table.style.format("{:.2f}")
        st.dataframe(styled_table, use_container_width=True)

    st.markdown("---")
    if st.button("🔙 " + _("Back to Prediction")):
        st.session_state.page = "predict"
        st.rerun()