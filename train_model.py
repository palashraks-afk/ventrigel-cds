import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report
)
import joblib

# Suppress convergence warnings for cleaner terminal output during training
warnings.filterwarnings("ignore")

print("="*60)
print("VENTRIGEL CLINICAL DECISION SUPPORT - AI TRAINING PIPELINE")
print("="*60)
time.sleep(1)

# ==========================================
# 1. DATA INGESTION & PREPROCESSING
# ==========================================
print("\n[1/6] Ingesting Synthetic Clinical Cohort...")
try:
    df = pd.read_csv('synthetic_ventrigel_cohort_v2.csv')
    print(f"  -> Successfully loaded {len(df)} patient records.")
except FileNotFoundError:
    print("  -> ERROR: 'synthetic_ventrigel_cohort_v2.csv' not found. Run generator first.")
    exit()

# Define Features (X) and Target (y)
# We MUST drop 'Patient_ID' (irrelevant) and 'Suitability_Score' (Data Leakage!)
X = df.drop(columns=['Patient_ID', 'Suitability_Score', 'Optimal_Candidate'])
y = df['Optimal_Candidate']

# Identify numeric vs categorical columns for the pipeline
categorical_cols = ['Cohort_Subgroup']
numeric_cols = X.columns.drop(categorical_cols).tolist()

print("\n[2/6] Building Feature Engineering Pipeline...")
# Standardize numeric features (vital for SVM and Neural Networks)
# One-hot encode categorical features (Early vs Late post-MI)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Create the standard train/test split (80% training, 20% unseen testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  -> Training Set: {len(X_train)} patients | Testing Set: {len(X_test)} patients.")

# ==========================================
# 3. INITIALIZING THE MODEL ARENA
# ==========================================
print("\n[3/6] Initializing Candidate Algorithms...")

# Define the models we want to compare. 
# class_weight='balanced' ensures the AI doesn't ignore the minority class
models = {
    "Logistic Regression (Baseline)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest (Clinical Standard)": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42),
    "Gradient Boosting (Ensemble)": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, class_weight='balanced', random_state=42),
    "Deep Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# ==========================================
# 4. CROSS-VALIDATION & TRAINING LOOP
# ==========================================
print("\n[4/6] Commencing K-Fold Cross-Validation & Model Training...")
print("  -> Pitting models against each other using 5-Fold CV.\n")

results = []
trained_pipelines = {}

# Strict clinical evaluation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"  Training: {name}...")
    
    # Bundle the preprocessing and the model into a single deployable pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # 1. Cross-Validation (Evaluates stability across different data slices)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # 2. Fit the pipeline on the full training data
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline
    
    # 3. Predict on the unseen Test Set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 4. Calculate Clinical Performance Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    # 5. Store Results
    results.append({
        'Model': name,
        'CV_ROC_AUC_Mean': cv_scores.mean(),
        'Test_Accuracy': acc,
        'Test_Precision': prec,
        'Test_Recall': rec,
        'Test_F1_Score': f1,
        'Test_ROC_AUC': roc
    })

# ==========================================
# 5. LEADERBOARD & CHAMPION SELECTION
# ==========================================
print("\n[5/6] Generating Performance Leaderboard...")
results_df = pd.DataFrame(results).sort_values(by='Test_ROC_AUC', ascending=False).reset_index(drop=True)

# Format the leaderboard for terminal display
display_df = results_df.copy()
for col in display_df.columns[1:]:
    display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'

print("\n" + "="*80)
print(display_df.to_string(index=False))
print("="*80 + "\n")

# Crown the Champion (The model with the highest ROC-AUC on the test set)
champion_name = results_df.iloc[0]['Model']
champion_pipeline = trained_pipelines[champion_name]

print(f"🏆 CHAMPION MODEL SELECTED: {champion_name} 🏆")

# Print detailed clinical report for the champion
y_pred_champ = champion_pipeline.predict(X_test)
print("\nChampion Model Detailed Classification Report:")
print(classification_report(y_test, y_pred_champ, target_names=["Sub-optimal (0)", "Optimal (1)"]))

# ==========================================
# 6. PIPELINE SERIALIZATION (DEPLOYMENT PREP)
# ==========================================
print("\n[6/6] Serializing Champion Model for Web App Deployment...")
model_filename = 'ventrigel_clinical_pipeline_v2.joblib'

# By saving the entire pipeline, the Streamlit app won't have to manually scale new user inputs.
# The pipeline handles raw input -> scaling -> encoding -> prediction automatically.
joblib.dump(champion_pipeline, model_filename)

print(f"  -> SUCCESS: Pipeline saved as '{model_filename}'")
print("\nTraining Sequence Complete. Ready for Phase 2 UI updates.")
print("="*60)