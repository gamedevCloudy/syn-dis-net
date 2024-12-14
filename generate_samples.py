import numpy as np
import pandas as pd
import os
import glob

# Parameters
N = 1000000          # total number of rows (adjust as needed)
CHUNK_SIZE = 10000      # number of rows per chunk
OUTPUT_DIR = "chunks"   # directory to store chunk files
FINAL_OUTPUT_FILE = "SynDisNet.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define disease families and diseases for variety
disease_families = {
    "Cardiovascular": ["Coronary Artery Disease (CAD)", "Hypertension", "Congestive Heart Failure (CHF)"],
    "Neurological": ["Alzheimer's Disease", "Parkinson's Disease", "Epilepsy"],
    "Respiratory": ["COPD", "Asthma", "Pneumonia"],
    "Oncological": ["Breast Cancer", "Lung Cancer", "Colon Cancer"],
    "Gastrointestinal": ["Irritable Bowel Syndrome (IBS)", "Crohn's Disease", "Hepatitis C"],
    "Metabolic/Endocrine": ["Diabetes Mellitus Type 2", "Hypothyroidism", "Hyperthyroidism"],
    "Musculoskeletal": ["Rheumatoid Arthritis (RA)", "Osteoarthritis (OA)", "Osteoporosis"],
    "Dermatological": ["Psoriasis", "Eczema", "Melanoma"],
    "Psychiatric": ["Major Depressive Disorder (MDD)", "Schizophrenia", "Bipolar Disorder"],
    "Infectious": ["HIV/AIDS", "Tuberculosis (TB)", "Malaria"]
}

# Baseline parameters if not defined
def get_baseline_params():
    return {
        "systolic_bp": ("normal", 120, 15),
        "diastolic_bp": ("normal", 80, 10),
        "chest_pain": ("categorical", {0:0.9,1:0.1}),
        "cholesterol": ("normal", 190, 40),
        "BMI": ("normal", 25, 3),
        "heart_rate": ("normal", 75, 12),
        "fatigue_severity": ("categorical", {0:0.6,1:0.3,2:0.1}),
        "cough_type": ("categorical", {0:0.85,1:0.1,2:0.05}),
        "blood_glucose": ("normal", 100, 20),
        "smoking_status": ("categorical", {0:0.6,1:0.2,2:0.2})
    }

disease_params = {
    "Coronary Artery Disease (CAD)": {
        "systolic_bp": ("normal", 130, 15),
        "diastolic_bp": ("normal", 80, 10),
        "chest_pain": ("categorical", {0:0.7,1:0.3}),
        "cholesterol": ("normal", 200, 40),
        "BMI": ("normal", 28, 4),
        "heart_rate": ("normal", 75, 10),
        "fatigue_severity": ("categorical", {0:0.5,1:0.3,2:0.2}),
        "cough_type": ("categorical", {0:0.8,1:0.1,2:0.1}),
        "blood_glucose": ("normal", 110, 20),
        "smoking_status": ("categorical", {0:0.5,1:0.3,2:0.2})
    },
    "Hypertension": {
        "systolic_bp": ("normal", 150, 20),
        "diastolic_bp": ("normal", 95, 15),
        "chest_pain": ("categorical", {0:0.8,1:0.2}),
        "cholesterol": ("normal", 210, 45),
        "BMI": ("normal", 30, 5),
        "heart_rate": ("normal", 80, 10),
        "fatigue_severity": ("categorical", {0:0.4,1:0.4,2:0.2}),
        "cough_type": ("categorical", {0:0.9,1:0.05,2:0.05}),
        "blood_glucose": ("normal", 120, 30),
        "smoking_status": ("categorical", {0:0.4,1:0.3,2:0.3})
    },
    "Congestive Heart Failure (CHF)": {
        "systolic_bp": ("normal", 120, 20),
        "diastolic_bp": ("normal", 75, 10),
        "chest_pain": ("categorical", {0:0.7,1:0.3}),
        "cholesterol": ("normal", 190, 35),
        "BMI": ("normal", 29, 4),
        "heart_rate": ("normal", 90, 15),
        "fatigue_severity": ("categorical", {0:0.4,1:0.4,2:0.2}),
        "cough_type": ("categorical", {0:0.7,1:0.15,2:0.15}),
        "blood_glucose": ("normal", 115, 25),
        "smoking_status": ("categorical", {0:0.5,1:0.25,2:0.25})
    }
}

# Fill in baseline for all not defined
for fam, dis_list in disease_families.items():
    for dis in dis_list:
        if dis not in disease_params:
            disease_params[dis] = get_baseline_params()

all_diseases = []
all_families = []
for fam, dis_list in disease_families.items():
    for d in dis_list:
        all_families.append(fam)
        all_diseases.append(d)

disease_probs = np.ones(len(all_diseases)) / len(all_diseases)
np.random.seed(42)

def generate_value(dist_type, *params):
    if dist_type == "normal":
        mean, std = params
        return np.random.normal(mean, std)
    elif dist_type == "categorical":
        prob_dict = params[0]
        categories = list(prob_dict.keys())
        probabilities = list(prob_dict.values())
        return np.random.choice(categories, p=probabilities)
    else:
        return None

columns = [
    "patient_id",
    "disease_family",
    "disease",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "BMI",
    "heart_rate",
    "fatigue_severity",
    "cough_type",
    "blood_glucose",
    "chest_pain",
    "smoking_status"
]

# Generate and write in chunks
num_chunks = N // CHUNK_SIZE
if N % CHUNK_SIZE != 0:
    num_chunks += 1

start_id = 1
for chunk_idx in range(num_chunks):
    chunk_size = CHUNK_SIZE
    if (start_id + CHUNK_SIZE - 1) > N:
        chunk_size = N - start_id + 1
    if chunk_size <= 0:
        break
    
    patient_ids = np.arange(start_id, start_id + chunk_size)
    selected_indices = np.random.choice(range(len(all_diseases)), size=chunk_size, p=disease_probs)
    selected_diseases = [all_diseases[i] for i in selected_indices]
    selected_families = [all_families[i] for i in selected_indices]

    # Preallocate arrays
    systolic_bp_col = np.empty(chunk_size)
    diastolic_bp_col = np.empty(chunk_size)
    cholesterol_col = np.empty(chunk_size)
    BMI_col = np.empty(chunk_size)
    heart_rate_col = np.empty(chunk_size)
    blood_glucose_col = np.empty(chunk_size)
    fatigue_severity_col = np.empty(chunk_size, dtype=int)
    cough_type_col = np.empty(chunk_size, dtype=int)
    chest_pain_col = np.empty(chunk_size, dtype=int)
    smoking_status_col = np.empty(chunk_size, dtype=int)

    for i, dis in enumerate(selected_diseases):
        params = disease_params[dis]
        systolic_bp_col[i] = generate_value(*params["systolic_bp"])
        diastolic_bp_col[i] = generate_value(*params["diastolic_bp"])
        cholesterol_col[i] = generate_value(*params["cholesterol"])
        BMI_col[i] = generate_value(*params["BMI"])
        heart_rate_col[i] = generate_value(*params["heart_rate"])
        blood_glucose_col[i] = generate_value(*params["blood_glucose"])
        fatigue_severity_col[i] = generate_value(*params["fatigue_severity"])
        cough_type_col[i] = generate_value(*params["cough_type"])
        chest_pain_col[i] = generate_value(*params["chest_pain"])
        smoking_status_col[i] = generate_value(*params["smoking_status"])

    df_chunk = pd.DataFrame({
        "patient_id": patient_ids,
        "disease_family": selected_families,
        "disease": selected_diseases,
        "systolic_bp": systolic_bp_col,
        "diastolic_bp": diastolic_bp_col,
        "cholesterol": cholesterol_col,
        "BMI": BMI_col,
        "heart_rate": heart_rate_col,
        "fatigue_severity": fatigue_severity_col,
        "cough_type": cough_type_col,
        "blood_glucose": blood_glucose_col,
        "chest_pain": chest_pain_col,
        "smoking_status": smoking_status_col
    })

    chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx+1}.csv")
    df_chunk.to_csv(chunk_file, index=False)
    
    start_id += chunk_size

print("Chunk generation completed.")

# Now, join all chunk files into one large file
chunk_files = glob.glob(os.path.join(OUTPUT_DIR, "chunk_*.csv"))
# Sort chunk files by name to maintain order (optional)
chunk_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

# To efficiently join, we can use a loop and write to the final file in append mode
# We'll write the header from the first file, then append the rest without headers.
with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    # Write the header from the first file
    with open(chunk_files[0], 'r', encoding='utf-8') as firstfile:
        header = firstfile.readline()
        outfile.write(header)
        # Write the rest of the first file
        for line in firstfile:
            outfile.write(line)
    
    # Append the rest of the files without headers
    for cf in chunk_files[1:]:
        with open(cf, 'r', encoding='utf-8') as f:
            # Skip header
            f.readline()
            for line in f:
                outfile.write(line)

print(f"All chunks have been combined into {FINAL_OUTPUT_FILE}.")
