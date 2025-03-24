import streamlit as st
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from utils import _, calculate_derived_features, save_patient_data, load_models, load_scaler_and_columns
import sqlite3

# التحقق من الاسم باللغة الإنجليزية فقط
def is_english(text):
    return bool(re.match('^[A-Za-z0-9\s]*$', text))

# التحقق من أن Patient ID يتكون من 6 أرقام فقط
def is_valid_patient_id(text):
    return bool(re.fullmatch(r'\d{6}', text))

# حفظ البيانات في قاعدة البيانات
def save_to_database(data):
    conn = sqlite3.connect("patient_data.db")
    cursor = conn.cursor()
    cursor.execute('''
         CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT,
        name TEXT,
        age INTEGER,
        weight REAL,
        height REAL,
        weeks_pregnant INTEGER,
        sbp REAL,
        dbp REAL,
        cr_se REAL,
        plt REAL,
        bun REAL,
        protein REAL,
        chol REAL,
        glu REAL,
        uric REAL,
        alk REAL,
        alt REAL,
        egfr REAL,
        bmi REAL,
        map_val REAL,
        plt_map_ratio REAL,
        height_m REAL,
        bun_cr_ratio REAL,
        result INTEGER  -- ✅ عدلناه من TEXT إلى INTEGER
    )
    ''')
    cursor.execute('''INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', data)
    conn.commit()
    conn.close()

# صفحة التنبؤ
def predict_page():
    st.title(_("preeclampsia prediction"))
    st.write(_("early_prediction"))


    # تحميل النماذج والمقياس
    scaler, saved_columns = load_scaler_and_columns()
    nn_model, xgb_model, lr_model = load_models()

    with st.form("prediction_form"):
        patient_id = st.text_input(_("Patient ID"), key="patient_id")
        name = st.text_input(_("Patient Name"), key="patient_name")

        if patient_id and not is_valid_patient_id(patient_id):
            st.warning(_("Patient ID must be exactly 6 digits (numbers only)."))

        if name and not is_english(name):
            st.warning(_("Please enter the Patient Name in English only!"))

        age = st.number_input(_("Age"), min_value=18, max_value=100, value=25)
        weight = st.number_input(_("Weight"), min_value=40.0, max_value=200.0, value=70.0)
        height = st.number_input(_("Height"), min_value=140.0, max_value=200.0, value=170.0)
        weeks_pregnant = st.number_input(_("Weeks Pregnant"), min_value=0, max_value=52, value=20)

        st.subheader(_("Clinical Data"))
        sbp = st.number_input(_("SBP"), min_value=90.0, max_value=180.0, value=120.0)
        dbp = st.number_input(_("DBP"), min_value=50.0, max_value=120.0, value=80.0)
        cr_se = st.number_input(_("CR_SE"), min_value=0.1, max_value=3.0, value=1.0)
        plt = st.number_input(_("Platelets"), min_value=150.0, max_value=400.0, value=250.0)
        bun = st.number_input(_("BUN"), min_value=7.0, max_value=20.0, value=12.0)
        protein = st.number_input(_("Protein"), min_value=6.0, max_value=8.0, value=7.0)
        chol = st.number_input(_("Cholesterol"), min_value=100.0, max_value=300.0, value=200.0)
        glu = st.number_input(_("Glucose"), min_value=70.0, max_value=140.0, value=100.0)
        uric = st.number_input(_("Uric Acid"), min_value=3.0, max_value=8.0, value=5.0)
        alk = st.number_input(_("ALK"), min_value=30.0, max_value=120.0, value=60.0)
        alt = st.number_input(_("ALT"), min_value=0.0, max_value=40.0, value=15.0)

        predict_button = st.form_submit_button(_("Predict"))

    if predict_button:
        if not is_valid_patient_id(patient_id) or not is_english(name):
            st.error(_("Please enter a valid Patient ID (6 digits only) and make sure the name is in English."))
            return

        height_m, bmi, map_val, bun_cr_ratio, plt_map_ratio, eGFR = calculate_derived_features(
            weight, height, sbp, dbp, bun, cr_se, plt, age
        )

        input_data = pd.DataFrame([{
            'Age': age,
            'K': 0.7,
            'MAP': map_val,
            'BUN_CR_ratio': bun_cr_ratio,
            'PLT': plt,
            'Height_m': height_m,
            'PLT_MAP_ratio': plt_map_ratio,
            'BUN': bun,
            'Weight': weight,
            'BMI': bmi,
            'DBP': dbp,
            'CHOL': chol,
            'weeks of childbirth': weeks_pregnant,
            'eGFR': eGFR,
            'Glu': glu,
            'ALT': alt,
            'PROTEIN': protein,
            'Uric': uric,
            'SBP': sbp,
            'CR_SE': cr_se,
            'ALK': alk,
            'Height': height
        }], columns=saved_columns)

        imputer = SimpleImputer(strategy='mean')
        input_data_imputed = imputer.fit_transform(input_data)
        input_scaled = scaler.transform(input_data_imputed)

        # التنبؤ باستخدام النماذج
        nn_probs = nn_model.predict(input_scaled).flatten()
        xgb_probs = xgb_model.predict_proba(input_scaled)[:, 1]
        lr_probs = lr_model.predict_proba(input_scaled)[:, 1]
        ensemble_probs = (nn_probs + xgb_probs + lr_probs) / 3.0
        ensemble_pred = (ensemble_probs > 0.5).astype(int)

        result_en = "Preeclampsia detected" if ensemble_pred[0] == 1 else "No preeclampsia detected"

        # حفظ البيانات
        # ✅ هنا تخزن النتيجة كـ 0 أو 1
        save_to_database((patient_id, name, age, weight, height, weeks_pregnant,
                  sbp, dbp, cr_se, plt, bun, protein, chol, glu, uric, alk, alt,
                  eGFR, bmi, map_val, plt_map_ratio, height_m, bun_cr_ratio, int(ensemble_pred[0])))


        # عرض النتيجة في صفحة منفصلة
        st.session_state.result = int(ensemble_pred[0])
        st.session_state.derived_table = pd.DataFrame([{
            "Height (m)": height_m,
            "BMI": bmi,
            "MAP": map_val,
            "BUN/CR Ratio": bun_cr_ratio,
            "PLT/MAP Ratio": plt_map_ratio,
            "eGFR": eGFR
        }])
        st.session_state.patient_info = {
            "Patient ID": patient_id,
            "Name": name,
            "Age": age,
            "Weight": weight,
            "Height": height,
            "Weeks Pregnant": weeks_pregnant
        }

        st.session_state.page = "result"
        st.rerun()

    if st.button(_("Back to Home")):
        st.session_state.page = "home"
        st.rerun()