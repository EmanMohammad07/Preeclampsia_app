import streamlit as st
import pandas as pd
from utils import _, calculate_derived_features, save_patient_data, calculate_egfr
from sklearn.impute import SimpleImputer # Already imported in utils, but for clarity if this was standalone


def predict_page(scaler, saved_columns, nn_model, xgb_model, lr_model): # Accept models and scaler as arguments
    st.title(_("preeclampsia prediction"))
    st.write(_("early_prediction"))

    with st.form("prediction_form"):
        patient_id = st.text_input(_("Patient ID"))
        name = st.text_input(_("Patient Name")) # Corrected label
        age = st.number_input(_("Age"), min_value=18, value=25)
        weight = st.number_input(_("Weight"), min_value=40.0, value=70.0)
        height = st.number_input(_("Height"), min_value=140.0, value=170.0)
        weeks_pregnant = st.number_input(_("Weeks Pregnant"), min_value=0, value=20)

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
        if not patient_id:
            st.warning(_("Please enter a valid Patient ID!"))
        else:
            # حساب القيم المشتقة بما في ذلك eGFR
            height_m, bmi, map_val, bun_cr_ratio, plt_map_ratio, eGFR = calculate_derived_features(weight, height, sbp, dbp, bun, cr_se, plt, age)

            # عرض القيم المشتقة للمستخدم
            st.write(f"**Height in meters (Height_m):** {height_m:.2f}")
            st.write(f"**BMI:** {bmi:.2f}")
            st.write(f"**MAP:** {map_val:.2f}")
            st.write(f"**BUN/CR Ratio:** {bun_cr_ratio:.2f}")
            st.write(f"**PLT/MAP Ratio:** {plt_map_ratio:.2f}")
            st.write(f"**eGFR:** {eGFR:.2f}")

            # إعداد البيانات للتنبؤ
            input_data = pd.DataFrame([{
                'Age': age,
                'K': 0.7, # إضافة ميزة 'K' بالقيمة 0.7
                'MAP': map_val,
                'BUN_CR_ratio': bun_cr_ratio,
                'PLT': plt,
                'Height_m': height_m,
                'PLT_MAP_ratio': plt_map_ratio,
                'BUN': bun,
                'Weight': weight,
                'BMI': bmi,
                'DBP': dbp,
                'CHOL': chol, # استخدام الاسم المختصر "CHOL"
                'weeks of childbirth': weeks_pregnant, # استخدام الاسم "weeks of childbirth"
                'eGFR': eGFR,
                'Glu': glu, # استخدام الاسم المختصر "Glu"
                'ALT': alt,
                'PROTEIN': protein, # استخدام الاسم المختصر "PROTEIN"
                'Uric': uric, # استخدام الاسم المختصر "Uric"
                'SBP': sbp,
                'CR_SE': cr_se,
                'ALK': alk,
                'Height': height # استخدام الاسم "Height"
            }], columns=saved_columns) # استخدام saved_columns المعدلة

            # استبدال القيم المفقودة باستخدام SimpleImputer
            imputer = SimpleImputer(strategy='mean')  # استبدال القيم المفقودة بالمتوسط
            input_data_imputed = imputer.fit_transform(input_data)  # استبدال القيم المفقودة

            # تحويل البيانات المفقودة بعد المعالجة
            input_scaled = scaler.transform(input_data_imputed)

            # Ensemble prediction
            nn_probs = nn_model.predict(input_scaled).flatten()
            xgb_probs = xgb_model.predict_proba(input_scaled)[:, 1]
            lr_probs = lr_model.predict_proba(input_scaled)[:, 1]
            ensemble_probs = (nn_probs + xgb_probs + lr_probs) / 3.0
            best_thresh = 0.5  # or the best threshold that you got from your model training.
            ensemble_pred = (ensemble_probs > best_thresh).astype(int)

            # تحديد النتيجة
            result = _("Preeclampsia detected") if ensemble_pred[0] == 1 else _("No preeclampsia detected")

            # رسالة التوصية بناءً على النتيجة
            recommendation = _("We recommend immediate medical attention.") if result == _("Preeclampsia detected") else _("No immediate medical concerns. Continue monitoring health regularly.")

            # حفظ البيانات المدخلة مع التنبؤ في patient_data.csv
            save_patient_data(patient_id, name, age, weight, height, weeks_pregnant, sbp, dbp, cr_se, plt, bun, protein, chol, glu, uric, alk, alt, eGFR, bmi, map_val, plt_map_ratio, height_m, bun_cr_ratio, result)

            # عرض النتيجة
            st.write(f"**Prediction Result:** {result}")
            st.write(f"**Recommendation:** {recommendation}")


    if st.button(_("Back to Home")):
        st.session_state.page = "home"
        st.rerun()

if __name__ == '__main__':
    # This is just for testing, when running standalone, it won't have scaler/models
    st.error("This page needs to be run from the main app.py so models and scaler are loaded.")