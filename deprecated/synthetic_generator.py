import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & TRIAL HYPERPARAMETERS
# ==========================================
N_PATIENTS = 2000  # Scaled up for robust machine learning training
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Initializing Advanced VentriGel Clinical Simulation Engine...")
print(f"Target Cohort Size: {N_PATIENTS} synthetic patient profiles.\n")

# ==========================================
# 2. STRATIFIED POPULATION GENERATION
# ==========================================
# Split cohort into early (<12 months) and late (>12 months) post-MI groups
# based on the study's primary stratifications.
n_early = int(N_PATIENTS * 0.45) # ~45% <12 months
n_late = N_PATIENTS - n_early     # ~55% >12 months

def generate_subcohort(n, time_bracket):
    # Demographics
    age = np.random.normal(58.5, 9.2, n).clip(30, 80)
    
    if time_bracket == 'early':
        days = np.random.uniform(30, 364, n)
        ef_mean, ef_std = 37.7, 3.6
        scar_mean, scar_std = 25.9, 1.6
        lvesv_mean = 156.8
    else:
        days = np.random.uniform(365, 1460, n)
        ef_mean, ef_std = 36.6, 3.3
        scar_mean, scar_std = 27.9, 2.9
        lvesv_mean = 142.3

    # Core Cardiac Mechanics (Online Table 7 parameters)
    ef = np.random.normal(ef_mean, ef_std, n)
    scar = np.random.normal(scar_mean, scar_std, n)
    
    # Ventricular Volumes & Mass
    lvesv = np.random.normal(lvesv_mean, 20.0, n)
    lvedv = lvesv + np.random.normal(80.0, 15.0, n) # Derived stroke volume relationship
    viable_mass = np.random.normal(114.2, 12.0, n)
    
    return pd.DataFrame({
        'Age': age,
        'Days_Post_MI': days,
        'Baseline_EF_Percent': ef,
        'Scar_Mass_Percent': scar,
        'LVESV_mL': lvesv,
        'LVEDV_mL': lvedv,
        'Viable_Mass_g': viable_mass,
        'Cohort_Subgroup': time_bracket
    })

df_early = generate_subcohort(n_early, 'early')
df_late = generate_subcohort(n_late, 'late')
df = pd.concat([df_early, df_late], ignore_index=True)

# ==========================================
# 3. BIOLOGICAL CROSS-TALK & MULTIVARIATE ADJUSTMENTS
# ==========================================
# Calculate a physiological penalty index based on ejection fraction and volume
ef_penalty = (37.1 - df['Baseline_EF_Percent']) / 2.3

# Biomarkers: BNP (Online Table 8: Mean 294.8 pg/mL)
bnp = np.random.normal(294.8, 86.4, N_PATIENTS) + (ef_penalty * 35.0)
df['BNP_Level_pgml'] = bnp.clip(50, 1500)

# Inflammatory Markers: C-Reactive Protein (Online Table 4: Mean 0.7 mg/L)
crp = np.random.normal(0.7, 0.4, N_PATIENTS) + (np.abs(ef_penalty) * 0.1)
df['CRP_Level_mgl'] = crp.clip(0.1, 10.0)

# Functional Capacity: 6-Minute Walk Test (Online Table 5: Mean 429.4 meters)
walk = np.random.normal(429.4, 28.0, N_PATIENTS) - (ef_penalty * 18.0)
df['Walk_6_Min_meters'] = walk.clip(150, 700)

# NYHA Heart Failure Class distribution (Classes I, II, III)
nyha_probs = np.random.uniform(0, 1, N_PATIENTS)
df['NYHA_Class'] = np.where(nyha_probs < 0.25, 1, np.where(nyha_probs < 0.75, 2, 3))

# ==========================================
# 4. PHARMACOTHERAPY PROFILES (CONCOMITANT MEDS)
# ==========================================
# Simulating real-world Guideline-Directed Medical Therapy (GDMT) from trial logs
df['Med_BetaBlocker'] = np.random.choice([0, 1], size=N_PATIENTS, p=[0.27, 0.73]) # Carvedilol/Metoprolol (~73%)
df['Med_ACE_ARB'] = np.random.choice([0, 1], size=N_PATIENTS, p=[0.13, 0.87])     # ACEi/ARBs (~87%)
df['Med_Statin'] = np.random.choice([0, 1], size=N_PATIENTS, p=[0.13, 0.87])      # Atorvastatin/Rosuvastatin (~87%)
df['Med_Antiplatelet'] = 1 # 100% adherence to antiplatelet therapy in trial

# ==========================================
# 5. ADVANCED CLINICAL SCORING & CANDIDACY RULE ENGINE
# ==========================================
def calculate_suitability_and_label(row):
    score = 100.0
    optimal = 1
    
    # Hard Protocol Exclusion Filters
    if not (30 <= row['Age'] <= 75): 
        optimal = 0
        score -= 40
    if not (25.0 <= row['Baseline_EF_Percent'] <= 45.0): 
        optimal = 0
        score -= 50
    if not (60 <= row['Days_Post_MI'] <= 1095): 
        optimal = 0
        score -= 45
        
    # Soft Risk Penalties (Nuanced Clinical Decision Support)
    if row['BNP_Level_pgml'] > 600: score -= 15
    if row['Walk_6_Min_meters'] < 300: score -= 15
    if row['NYHA_Class'] == 3: score -= 10
    if row['CRP_Level_mgl'] > 3.0: score -= 10
    
    # If cumulative penalties drop score below threshold, classify as sub-optimal
    if score < 60:
        optimal = 0
        
    return pd.Series([max(0.0, round(score, 1)), optimal])

df[['Suitability_Score', 'Optimal_Candidate']] = df.apply(calculate_suitability_and_label, axis=1)

# ==========================================
# 6. DATA CLEANING, ROUNDING & EXPORT
# ==========================================
# Final boundary clipping for physiological realism
df['Baseline_EF_Percent'] = df['Baseline_EF_Percent'].clip(15.0, 55.0).round(1)
df['Scar_Mass_Percent'] = df['Scar_Mass_Percent'].clip(5.0, 50.0).round(1)
df['LVESV_mL'] = df['LVESV_mL'].round(1)
df['LVEDV_mL'] = df['LVEDV_mL'].round(1)
df['Viable_Mass_g'] = df['Viable_Mass_g'].round(1)
df['Days_Post_MI'] = df['Days_Post_MI'].round(0)
df['Age'] = df['Age'].round(0)
df['BNP_Level_pgml'] = df['BNP_Level_pgml'].round(1)
df['CRP_Level_mgl'] = df['CRP_Level_mgl'].round(2)
df['Walk_6_Min_meters'] = df['Walk_6_Min_meters'].round(1)

# Assign Patient IDs
df.insert(0, 'Patient_ID', [f"VG-SYN-{i:04d}" for i in range(1, N_PATIENTS + 1)])

# Export to CSV
output_filename = "synthetic_ventrigel_cohort_v2.csv"
df.to_csv(output_filename, index=False)

print(f"Pipeline Execution Successful!")
print(f"File Saved: {output_filename}")
print(f"Total Records Generated: {len(df)}")
print("\nFinal Target Distribution:")
dist = df['Optimal_Candidate'].value_counts(normalize=True) * 100
print(f"  * Optimal Candidates (1): {dist.get(1, 0):.1f}%")
print(f"  * Sub-optimal Candidates (0): {dist.get(0, 0):.1f}%")