import streamlit as st
from utils import _

def preeclampsia_info_page():
    st.title(_("What is Preeclampsia?"))
    st.write(_(
        "Preeclampsia is a serious medical condition that can occur after the 20th week of pregnancy. "
        "It is characterized by elevated blood pressure and the presence of protein in the urine or signs of organ dysfunction, "
        "particularly in the liver or kidneys. If left untreated, it may lead to life-threatening complications for both mother and baby."
    ))

    st.header(_("Signs and Symptoms of Preeclampsia"))
    st.write(_("Symptoms can range from mild to severe, and may include:"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("Elevated blood pressure (≥ 140/90 mmHg)"),
        _("Unusual swelling of the face, hands, or feet"),
        _("Severe headache unresponsive to medication"),
        _("Visual disturbances (blurred vision, flashing lights)"),
        _("Pain in the upper right abdomen"),
        _("Shortness of breath or chest tightness"),
        _("Decreased urine output or dark-colored urine")
    ))

    st.header(_("When to Intervene Medically"))
    st.write(_("Immediate medical intervention is recommended if any of the following criteria are present:"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("Blood pressure ≥ 140/90 mmHg on two readings at least 4 hours apart"),
        _("Proteinuria ≥ 300 mg/24h or Protein/Creatinine ratio ≥ 0.3"),
        _("Persistent, severe headache"),
        _("Neurological or visual disturbances"),
        _("Platelet count < 100,000/μL"),
        _("Elevated creatinine ≥ 1.1 mg/dL or doubling of baseline"),
        _("Elevated liver enzymes (ALT/AST)"),
        _("Pulmonary edema or unexplained dyspnea"),
        _("Signs of fetal compromise (growth restriction, oligohydramnios)")
    ))

    st.header(_("Classification of Preeclampsia"))

    st.subheader(_("1. Mild Preeclampsia"))
    st.markdown("""
        - {}
        - {}
        - {}
    """.format(
        _("Elevated blood pressure without severe symptoms"),
        _("Normal liver and kidney function"),
        _("No neurological signs")
    ))

    st.subheader(_("2. Severe Preeclampsia"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("BP ≥ 160/110 mmHg"),
        _("Visual or neurological disturbances"),
        _("Organ dysfunction (liver, kidney, or platelet issues)"),
        _("Clinical deterioration requiring hospitalization or early delivery")
    ))

    st.subheader(_("3. Early-Onset Preeclampsia (< 34 weeks)"))
    st.markdown(_("Associated with higher risk; requires close monitoring and delivery planning."))

    st.subheader(_("4. Late-Onset Preeclampsia (≥ 34 weeks)"))
    st.markdown(_("Often less severe but still warrants close surveillance."))

    st.header(_("Diagnosis"))
    st.write(_("Diagnostic workup includes:"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("Serial blood pressure measurements"),
        _("Urinalysis for protein"),
        _("Kidney function tests (Creatinine, BUN)"),
        _("Liver function tests (ALT, AST)"),
        _("Complete blood count (CBC)"),
        _("Fetal assessments (Ultrasound, Doppler flow studies)"),
        _("Additional markers: eGFR, BUN/Creatinine ratio, PLT/MAP ratio")
    ))

    st.header(_("Potential Complications"))
    st.write(_("If not managed, preeclampsia can lead to:"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("Placental abruption"),
        _("Intrauterine growth restriction (IUGR)"),
        _("Preterm birth"),
        _("HELLP syndrome"),
        _("Eclampsia (seizures)"),
        _("Renal or hepatic failure"),
        _("Maternal and/or fetal death in severe cases")
    ))

    st.header(_("Prevention and Monitoring"))
    st.write(_("While preeclampsia cannot always be prevented, risk can be reduced through:"))
    st.markdown("""
        - {}
        - {}
        - {}
        - {}
        - {}
        - {}
    """.format(
        _("Regular antenatal blood pressure monitoring"),
        _("Low-dose aspirin (81 mg daily starting at 12 weeks) in high-risk patients"),
        _("Managing chronic conditions like hypertension or diabetes"),
        _("Health education and balanced nutrition"),
        _("Light physical activity (if no contraindications)"),
        _("Monitoring weight gain and fluid retention")
    ))

    st.header(_("References and Resources"))
    st.markdown("""
        - [ACOG – American College of Obstetricians and Gynecologists](https://www.acog.org)
        - [WHO – World Health Organization](https://www.who.int)
        - [Mayo Clinic – Preeclampsia Overview](https://www.mayoclinic.org)
    """)

    # Back button
    if st.button(_("Back to Home Page")):
        st.session_state.page = "home"
        st.rerun()