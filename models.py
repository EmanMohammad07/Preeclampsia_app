import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
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

# -------------------------------------------
# 1️⃣ Load and Prepare Data
# -------------------------------------------
df = pd.read_csv('df_ADASYN_merge.csv')

# Separate Features and Target
X = df.drop(columns=['Preeclampsia Status'])  # Features
y = df['Preeclampsia Status']  # Target (1 = Preeclampsia, 0 = No Preeclampsia)

# حفظ أسماء الأعمدة قبل التحجيم
column_names = X.columns.tolist()
scaler_columns_path = os.path.join(assets_dir, 'scaler_columns.json')
with open(scaler_columns_path, 'w') as f:
    json.dump(column_names, f)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------
# 2️⃣ Stratified K-Fold Cross-Validation Setup
# -------------------------------------------
k_folds = 5  # Number of folds
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store results for each fold
f1_scores = []
best_thresholds = []

# -------------------------------------------
# 3️⃣ Define Base Models (Neural Network, XGBoost, Logistic Regression)
# -------------------------------------------
def create_nn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary Classification Output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

# -------------------------------------------
# 4️⃣ Train & Evaluate Stacking Model using K-Fold Cross-Validation
# -------------------------------------------
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\n🔹 Fold {fold+1}/{k_folds}")

    # Split into train and test for this fold
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Split train further into base-training (80%) and meta-training (20%)
    X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Train Neural Network
    input_shape = (X_train_base.shape[1],)
    nn_model = create_nn_model(input_shape)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(assets_dir, f'best_nn_model_fold_{fold+1}.h5'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("Training Neural Network Model...")

    nn_model.fit(X_train_base, y_train_base, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)

    # Train XGBoost Model
    print("\nTraining XGBoost Model...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_base, y_train_base)

    # Train Logistic Regression Model
    print("\nTraining Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_base, y_train_base)

    # Generate Meta-Features
    nn_meta_probs = nn_model.predict(X_train_meta).flatten()
    xgb_meta_probs = xgb_model.predict_proba(X_train_meta)[:, 1]
    lr_meta_probs = lr_model.predict_proba(X_train_meta)[:, 1]

    X_meta_features = np.column_stack([nn_meta_probs, xgb_meta_probs, lr_meta_probs])

    # Train Meta-Learner
    print("\nTraining Meta-Learner...")
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_meta_features, y_train_meta)

    # Predict on Test Set
    nn_test_probs = nn_model.predict(X_test).flatten()
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    lr_test_probs = lr_model.predict_proba(X_test)[:, 1]

    X_test_meta_features = np.column_stack([nn_test_probs, xgb_test_probs, lr_test_probs])
    meta_test_probs = meta_learner.predict_proba(X_test_meta_features)[:, 1]

    # Find Best Threshold (Grid Search)
    beta = 2
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_thresh = max(thresholds, key=lambda t: fbeta_score(y_test, (meta_test_probs > t).astype(int), beta=beta))
    best_thresholds.append(best_thresh)

    # Apply Best Threshold
    ensemble_pred = (meta_test_probs > best_thresh).astype(int)

    # Evaluate Model
    fold_f1 = f1_score(y_test, ensemble_pred)
    f1_scores.append(fold_f1)

    print("\n🔹 Stacking Ensemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))

    print("\n🔹 Stacking Ensemble Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))

    print(f"\n✅ Fold {fold+1} - Best Threshold: {best_thresh:.2f}, F1 Score: {fold_f1:.4f}")

# -------------------------------------------
# 5️⃣ Final Model Performance Summary
# -------------------------------------------
print("\n=========================")
print("📌 Cross-Validation Summary")
print("=========================")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
print(f"Thresholds Used: {best_thresholds}")

# -------------------------------------------
# 6️⃣ Save Final Models & Preprocessing
# -------------------------------------------
print("\nSaving Models and Preprocessing Data...")
with open(os.path.join(assets_dir, 'preeclampsia_scaler.pkl'), 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open(os.path.join(assets_dir, 'scaler_columns.json'), 'w') as f:
    json.dump(X.columns.tolist(), f)

# Save Base Models & Meta-Learner
nn_model.save(os.path.join(assets_dir, 'preeclampsia_nn_model.keras'))
xgb_model.save_model(os.path.join(assets_dir, 'preeclampsia_xgb_model.json'))
with open(os.path.join(assets_dir, 'preeclampsia_lr_model.pkl'), 'wb') as lr_file:
    pickle.dump(lr_model, lr_file)

with open(os.path.join(assets_dir, 'preeclampsia_meta_learner.pkl'), 'wb') as meta_file:
    pickle.dump(meta_learner, meta_file)

print("\n✅ All Models and Preprocessing Files Saved Successfully!")
