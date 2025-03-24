import streamlit as st
import pandas as pd
from utils import _

def result_page():
    st.title(_("Prediction Result"))

    result = st.session_state.get("result", None)
    derived_table = st.session_state.get("derived_table", None)
    patient_info = st.session_state.get("patient_info", {})

    if result is None:
        st.error(_("No prediction result available. Please go back and run the prediction first."))
        if st.button(_("Back to Prediction")):
            st.session_state.page = "predict"
            st.rerun()
        return

    # Display prediction result
    if result == 0:
        st.success(_("No preeclampsia detected"))  # ترجمة نتيجة التنبؤ
        st.info(_("No immediate medical concerns. Continue monitoring health regularly."))
    else:
        st.error(_("Preeclampsia detected"))  # ترجمة نتيجة التنبؤ
        st.warning(_("We recommend immediate medical attention."))

    # --- Medical Recommendations Section ---
    st.markdown("---")

    if result == 0:
        st.markdown(f"## {_('Recommended clinical actions when preeclampsia is not diagnosed')}")
        st.markdown(f"""
- {_('Continue routine antenatal care.')} 
- {_('Monitor blood pressure and perform regular urine protein analysis.')} 
- {_('Educate the patient about warning signs such as:')} 
    - {_('Persistent headache')} 
    - {_('Vision changes')} 
    - {_('Upper abdominal pain')} 
    - {_('Unusual swelling')} 
- {_('Re-evaluation is recommended if new symptoms arise or if vital signs or lab results change.')}
""")
    else:
        st.markdown(f"## {_('Recommended clinical management after diagnosing preeclampsia')}")
        st.markdown(f"""
- {_('Immediate clinical evaluation to determine whether hospitalization is needed.')} 
- {_('Request additional tests, including:')} 
    - {_('Liver and kidney function tests')} 
    - {_('Complete blood count (CBC)')} 
- {_('Assess fetal growth and well-being using ultrasound and Doppler studies.')} 
- {_('Determine appropriate timing of delivery based on case severity and gestational age.')} 
- {_('Refer to a specialized care center if necessary.')}
""")

    # --- Derived Clinical Features Table ---
    if derived_table is not None:
        st.markdown("---")
        st.markdown(f"## {_('Derived Clinical Features')}")
        styled_table = derived_table.style.format("{:.2f}")
        st.dataframe(styled_table, use_container_width=True)

    # Back button
    st.markdown("---")
    if st.button(_("Back to Prediction")):
        st.session_state.page = "predict"
        st.rerun()