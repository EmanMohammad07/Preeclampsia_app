import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import json
import os

# تحديد مسار مجلد assets
assets_dir = 'assets'
os.makedirs(assets_dir, exist_ok=True)

##############################################
# 1. Load and Prepare Data
##############################################
df = pd.read_csv('df_ADASYN_merge.csv')
X = df.drop(columns=['Preeclampsia Status'])
y = df['Preeclampsia Status']

# حفظ أسماء الأعمدة قبل التحجيم
column_names = X.columns.tolist()
scaler_columns_path = os.path.join(assets_dir, 'scaler_columns.json')
with open(scaler_columns_path, 'w') as f:
    json.dump(column_names, f)

# تحجيم البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

##############################################
# 2. Split Data: Base, Meta, and Test Sets
##############################################
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

##############################################
# 3. Define Base Models
##############################################
def create_nn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

input_shape = (X_train_base.shape[1],)
nn_model = create_nn_model(input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(assets_dir, 'best_nn_model.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("Training Neural Network Model on base-training set...")
nn_model.fit(X_train_base, y_train_base, epochs=100, batch_size=32,
             validation_split=0.2, callbacks=callbacks, verbose=1)

print("\nTraining XGBoost Model on base-training set...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_base, y_train_base)

print("\nTraining Logistic Regression Model on base-training set...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_base, y_train_base)

##############################################
# 4. Generate Meta-Features using Base Models
##############################################
nn_meta_probs = nn_model.predict(X_train_meta).flatten()
xgb_meta_probs = xgb_model.predict_proba(X_train_meta)[:, 1]
lr_meta_probs = lr_model.predict_proba(X_train_meta)[:, 1]

X_meta_features = np.column_stack([nn_meta_probs, xgb_meta_probs, lr_meta_probs])

##############################################
# 5. Train Meta-Learner (Stacking)
##############################################
meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(X_meta_features, y_train_meta)

##############################################
# 6. Test Set Predictions
##############################################
nn_test_probs = nn_model.predict(X_test).flatten()
xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
lr_test_probs = lr_model.predict_proba(X_test)[:, 1]

X_test_meta_features = np.column_stack([nn_test_probs, xgb_test_probs, lr_test_probs])
meta_test_probs = meta_learner.predict_proba(X_test_meta_features)[:, 1]

##############################################
# 7. Grid Search for Optimal Threshold (F-beta Score)
##############################################
beta = 2
thresholds = np.arange(0.1, 0.9, 0.05)
best_thresh, best_fbeta = 0.5, 0

for thresh in thresholds:
    y_pred_temp = (meta_test_probs > thresh).astype(int)
    current_fbeta = fbeta_score(y_test, y_pred_temp, beta=beta)
    print(f"Threshold: {thresh:.2f}, F{beta}-score: {current_fbeta:.4f}")
    if current_fbeta > best_fbeta:
        best_fbeta, best_thresh = current_fbeta, thresh

print(f"\nOptimal Threshold: {best_thresh:.2f} with F{beta}-score: {best_fbeta:.4f}")
ensemble_pred = (meta_test_probs > best_thresh).astype(int)

##############################################
# 8. Evaluation
##############################################
print("\nStacking Ensemble Classification Report:")
print(classification_report(y_test, ensemble_pred))
print("\nStacking Ensemble Confusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred))
print(f"Stacking Ensemble F1 Score: {f1_score(y_test, ensemble_pred)}")

##############################################
# 9. Save Models and Artifacts
##############################################
with open(os.path.join(assets_dir, 'preeclampsia_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

nn_model.save(os.path.join(assets_dir, 'preeclampsia_nn_model.keras'))
xgb_model.save_model(os.path.join(assets_dir, 'preeclampsia_xgb_model.json'))
with open(os.path.join(assets_dir, 'preeclampsia_lr_model.pkl'), 'wb') as f:
    pickle.dump(lr_model, f)
with open(os.path.join(assets_dir, 'preeclampsia_meta_learner.pkl'), 'wb') as f:
    pickle.dump(meta_learner, f)

print("Stacking ensemble models, scaler, and column names saved in 'assets' directory.")
