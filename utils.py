import pickle
import json
import os
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import streamlit as st
import traceback  # Make sure traceback is imported

# Define assets path relative to utils.py
base_path = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(base_path, "assets")
locales_path = os.path.join(base_path, "locales")


# Load Scaler and columns
def load_scaler_and_columns():
    try:
        with open(os.path.join(assets_path, 'preeclampsia_scaler.pkl'), 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # تحميل قائمة الأعمدة من ملف json (مفضل لسهولة التعديل)
        with open(os.path.join(assets_path, 'scaler_columns.json'), 'r') as f:
            saved_columns = json.load(f)

        return scaler, saved_columns
    except FileNotFoundError as e:
        st.error(f"Error loading scaler or columns: {e}")
        st.stop()
        return None, None

# Load Models
def load_models():
    try:
        nn_model = tf.keras.models.load_model(os.path.join(assets_path, 'preeclampsia_nn_model.keras'))
        xgb_model = XGBClassifier()
        xgb_model.load_model(os.path.join(assets_path, 'preeclampsia_xgb_model.json'))
        with open(os.path.join(assets_path, 'preeclampsia_lr_model.pkl'), 'rb') as lr_file:
            lr_model = pickle.load(lr_file)
        return nn_model, xgb_model, lr_model
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.stop()
        return None, None, None

# دالة لحساب eGFR بناءً على المعادلة التي زودتني بها
def calculate_egfr(CR_SE, Age):
    k = 0.7
    alpha = -0.329
    factor_female = 1.018
    eGFR = 141 * min(CR_SE / k, 1)**alpha * max(CR_SE / k, 1)**(-1.209) * (0.993**Age) * factor_female
    return eGFR

# دالة لحساب الميزات المشتقة
def calculate_derived_features(weight, height, sbp, dbp, bun, cr_se, plt, age):
    height_m = height / 100  # تحويل الطول إلى متر
    bmi = weight / (height_m ** 2) if height_m != 0 else 0  # حساب الـ BMI
    map_val = (1/3) * sbp + (2/3) * dbp  # حساب MAP
    bun_cr_ratio = bun / cr_se if cr_se != 0 else 0  # حساب BUN/CR Ratio
    plt_map_ratio = plt / map_val if map_val != 0 else 0  # حساب PLT/MAP Ratio
    eGFR = calculate_egfr(cr_se, age)  # حساب eGFR
    return height_m, bmi, map_val, bun_cr_ratio, plt_map_ratio, eGFR

# حفظ البيانات المدخلة في patient_data.csv
def save_patient_data(patient_id, name, age, weight, height, weeks_pregnant, sbp, dbp, cr_se, plt, bun, protein, chol, glu, uric, alk, alt, eGFR, bmi, map_val, plt_map_ratio, height_m, bun_cr_ratio, prediction):
    new_data = pd.DataFrame([{
        'Patient ID': patient_id,
        'Name': name,
        'Age': age,
        'Weight': weight,
        'Height': height,
        'Weeks Pregnant': weeks_pregnant,
        'SBP': sbp,
        'DBP': dbp,
        'CR_SE': cr_se,
        'Platelets': plt,
        'BUN': bun,
        'Protein': protein,
        'Cholesterol': chol,
        'Glucose': glu,
        'Uric Acid': uric,
        'ALK': alk,
        'ALT': alt,
        'eGFR': eGFR,
        'BMI': bmi,
        'MAP': map_val,
        'PLT_MAP_ratio': plt_map_ratio,
        'Height_m': height_m,
        'BUN_CR_ratio': bun_cr_ratio,
        'Prediction': prediction
    }])

    if os.path.exists(os.path.join(assets_path, 'patient_data.csv')):
        patient_data = pd.read_csv(os.path.join(assets_path, 'patient_data.csv'))
        patient_data = pd.concat([patient_data, new_data], ignore_index=True)
        patient_data.to_csv(os.path.join(assets_path, 'patient_data.csv'), index=False)
    else:
        new_data.to_csv(os.path.join(assets_path, 'patient_data.csv'), index=False)

# تحميل البيانات من ملف CSV
def load_patient_data():
    try:
        return pd.read_csv(os.path.join(assets_path, 'patient_data.csv'))
    except FileNotFoundError:
        st.warning("patient_data.csv not found. Please ensure you save patient data first.")
        return None

# البحث عن المريض بواسطة Patient ID
def search_patient_by_id(patient_id, patient_data):
    result = patient_data[patient_data['Patient ID'] == int(patient_id)]
    if not result.empty:
        return result
    return None

# تحميل ملفات الترجمة من مجلد locales
def load_translation(language):
    locale_file = os.path.join(locales_path, f"{language}.json")
    print(f"DEBUG: Attempting to load translation file: {locale_file}") # Debug print

    if os.path.exists(locale_file):
        try:
            with open(locale_file, "r", encoding="utf-8") as file:
                translation_data = json.load(file)
                print(f"DEBUG: Successfully loaded translations for language: {language}") # Debug print
                return translation_data
        except json.JSONDecodeError as e:
            st.error(f"⚠ Error in {language}.json file: {e}")
            print(f"DEBUG: JSONDecodeError for {language} file: {e}") # Debug print
            return {}
        except Exception as e:  # Catch ANY other errors during file loading/parsing
            st.error(f"⚠ Unexpected error loading {language}.json: {e}")
            print(f"DEBUG: Unexpected error loading {language}.json: {e}") # Debug print
            traceback.print_exc()  # Print full traceback to console for deeper debugging
            return {}
    else:
        st.error(f"⚠ Translation file {language}.json not found at {locale_file}!")
        print(f"DEBUG: FileNotFoundError: {locale_file} not found") # Debug print
        return {}

# دالة الترجمة - We will initialize translations in app.py
def _(text):
    # This will be set in app.py using session_state
    translations = st.session_state.get('translations', {})
    return translations.get(text, text)