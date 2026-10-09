"""
WellCare AI - Preprocessing & Data Management Module
===================================================
Handles input validation, CSV multi-user persistence, missing values,
feature engineering, Scikit-Learn ColumnTransformer preprocessing pipelines,
and data separation (X, y) without leaking personal identifiers.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

import os
import random
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scoring import calculate_wellness_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "wellcare_users.csv")

# Standard CSV Columns
CSV_COLUMNS = [
    "User_ID",
    "Name",
    "Age",
    "Gender",
    "Height",
    "Weight",
    "Steps",
    "Exercise_Duration",
    "Exercise_Frequency",
    "Sleep_Hours",
    "Water_Intake",
    "Screen_Time",
    "Work_Study_Hours",
    "Meal_Regularity",
    "Stress_Level",
    "Mood_Level",
    "Energy_Level",
    "Smoking",
    "Alcohol",
    "Caffeine",
    "Concerns",
    "Wellness_Score",
    "Wellness_Category",
    "Timestamp",
]

# ML Feature Subsets (Excludes Personal Identifiers like User_ID and Name)
NUMERICAL_FEATURES = [
    "Age",
    "Height",
    "Weight",
    "Steps",
    "Exercise_Duration",
    "Exercise_Frequency",
    "Sleep_Hours",
    "Water_Intake",
    "Screen_Time",
    "Work_Study_Hours",
    "Stress_Level",
    "Mood_Level",
    "Energy_Level",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Meal_Regularity",
    "Smoking",
    "Alcohol",
    "Caffeine",
]

ALL_ML_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_VARIABLE = "Wellness_Category"


def ensure_data_directory() -> None:
    """Ensures the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_csv_exists() -> str:
    """Initializes the CSV file with standard headers if it does not already exist."""
    ensure_data_directory()
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        df_empty = pd.DataFrame(columns=CSV_COLUMNS)
        df_empty.to_csv(CSV_PATH, index=False)
    return CSV_PATH


def generate_next_user_id() -> str:
    """Generates the next sequential User_ID (e.g. WC1001, WC1002)."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty or "User_ID" not in df.columns:
            return "WC1001"
        valid_ids = []
        for uid in df["User_ID"].dropna().astype(str):
            if uid.startswith("WC") and uid[2:].isdigit():
                valid_ids.append(int(uid[2:]))
        if valid_ids:
            return f"WC{max(valid_ids) + 1}"
        return f"WC{len(df) + 1001}"
    except Exception:
        return f"WC{random.randint(1001, 9999)}"


def validate_user_input(raw_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validates and casts raw user form input safely.
    Returns (is_valid, error_message, clean_data).
    """
    try:
        clean = {}
        # 1. Personal Details
        clean["User_ID"] = str(raw_data.get("User_ID", "")).strip() or generate_next_user_id()
        clean["Name"] = str(raw_data.get("Name", "Anonymous User")).strip() or "Anonymous User"
        
        age = int(float(raw_data.get("Age", 25)))
        if not (10 <= age <= 120):
            return False, "Age must be between 10 and 120 years.", {}
        clean["Age"] = age

        gender = str(raw_data.get("Gender", "Prefer not to say")).strip().capitalize()
        clean["Gender"] = gender if gender in ["Male", "Female", "Other", "Non-binary"] else "Other"

        # 2. Physical Details
        height = float(raw_data.get("Height", 170))
        if not (50.0 <= height <= 250.0):
            return False, "Height must be between 50 cm and 250 cm.", {}
        clean["Height"] = round(height, 1)

        weight = float(raw_data.get("Weight", 65))
        if not (20.0 <= weight <= 300.0):
            return False, "Weight must be between 20 kg and 300 kg.", {}
        clean["Weight"] = round(weight, 1)

        steps = int(float(raw_data.get("Steps", 5000)))
        if not (0 <= steps <= 100000):
            return False, "Daily steps must be non-negative and realistic (0 - 100,000).", {}
        clean["Steps"] = steps

        ex_dur = float(raw_data.get("Exercise_Duration", 30))
        if not (0.0 <= ex_dur <= 600.0):
            return False, "Exercise duration must be between 0 and 600 minutes.", {}
        clean["Exercise_Duration"] = round(ex_dur, 1)

        ex_freq = int(float(raw_data.get("Exercise_Frequency", 3)))
        if not (0 <= ex_freq <= 7):
            return False, "Exercise frequency must be between 0 and 7 days per week.", {}
        clean["Exercise_Frequency"] = ex_freq

        # 3. Lifestyle
        sleep = float(raw_data.get("Sleep_Hours", 7.0))
        if not (0.0 <= sleep <= 24.0):
            return False, "Sleep hours must be between 0 and 24 hours.", {}
        clean["Sleep_Hours"] = round(sleep, 1)

        water = float(raw_data.get("Water_Intake", 2.5))
        if not (0.0 <= water <= 15.0):
            return False, "Water intake must be between 0 and 15 liters.", {}
        clean["Water_Intake"] = round(water, 1)

        screen = float(raw_data.get("Screen_Time", 5.0))
        if not (0.0 <= screen <= 24.0):
            return False, "Screen time must be between 0 and 24 hours.", {}
        clean["Screen_Time"] = round(screen, 1)

        work = float(raw_data.get("Work_Study_Hours", 8.0))
        if not (0.0 <= work <= 24.0):
            return False, "Work/study hours must be between 0 and 24 hours.", {}
        clean["Work_Study_Hours"] = round(work, 1)

        meal = str(raw_data.get("Meal_Regularity", "Regular")).strip().capitalize()
        clean["Meal_Regularity"] = meal if meal in ["Regular", "Moderate", "Irregular"] else "Regular"

        # 4. Mental/Emotional
        stress = int(float(raw_data.get("Stress_Level", 5)))
        if not (1 <= stress <= 10):
            return False, "Stress level must be on a scale of 1 to 10.", {}
        clean["Stress_Level"] = stress

        mood = int(float(raw_data.get("Mood_Level", 5)))
        if not (1 <= mood <= 10):
            return False, "Mood level must be on a scale of 1 to 10.", {}
        clean["Mood_Level"] = mood

        energy = int(float(raw_data.get("Energy_Level", 5)))
        if not (1 <= energy <= 10):
            return False, "Energy level must be on a scale of 1 to 10.", {}
        clean["Energy_Level"] = energy

        # 5. Habits & Concerns
        smoking = str(raw_data.get("Smoking", "No")).strip().capitalize()
        clean["Smoking"] = smoking if smoking in ["No", "Occasional", "Regular"] else "No"

        alcohol = str(raw_data.get("Alcohol", "No")).strip().capitalize()
        clean["Alcohol"] = alcohol if alcohol in ["No", "Occasional", "Regular"] else "No"

        caffeine = str(raw_data.get("Caffeine", "Moderate")).strip()
        clean["Caffeine"] = caffeine or "Moderate"

        clean["Concerns"] = str(raw_data.get("Concerns", "None")).strip() or "None"
        clean["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return True, None, clean

    except (ValueError, TypeError) as e:
        return False, f"Invalid numerical or formatting input: {str(e)}", {}


def save_user_record(clean_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Appends a new user record safely to wellcare_users.csv."""
    ensure_csv_exists()
    try:
        # Calculate score and category if not already calculated
        if "Wellness_Score" not in clean_data or "Wellness_Category" not in clean_data:
            scoring = calculate_wellness_score(clean_data)
            clean_data["Wellness_Score"] = scoring["Wellness_Score"]
            clean_data["Wellness_Category"] = scoring["Wellness_Category"]

        new_row = {col: clean_data.get(col, "") for col in CSV_COLUMNS}
        df_new = pd.DataFrame([new_row])

        # Thread/safe append
        if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
            df_new.to_csv(CSV_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(CSV_PATH, mode="w", header=True, index=False)

        return True, "User record saved successfully."
    except Exception as e:
        return False, f"Failed to save record: {str(e)}"


def load_all_users() -> pd.DataFrame:
    """Loads all records from CSV and handles types, duplicates, and missing values."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)

        # Drop pure duplicates
        df = df.drop_duplicates(subset=["User_ID"], keep="last")

        # Cast numerical columns safely
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Wellness_Score" in df.columns:
            df["Wellness_Score"] = pd.to_numeric(df["Wellness_Score"], errors="coerce")

        # Clean string/categoricals
        for col in CATEGORICAL_FEATURES + ["User_ID", "Name", "Concerns", "Wellness_Category"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single user record as a dict by User_ID."""
    df = load_all_users()
    if df.empty or "User_ID" not in df.columns:
        return None
    match = df[df["User_ID"].astype(str) == str(user_id)]
    if not match.empty:
        return match.iloc[-1].to_dict()
    return None


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - Numerical: SimpleImputer(median) + StandardScaler()
    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder()
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts features X and target y.
    Strictly excludes personal-identification columns (User_ID, Name, Timestamp, Concerns).
    """
    # Filter available ML features
    feature_cols = [c for c in ALL_ML_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_VARIABLE].copy()
    return X, y


def generate_seed_dataset(n: int = 60) -> int:
    """
    Generates n realistic synthetic wellness records and saves them to the CSV.
    Useful for initializing/bootstrapping training and visualization when no data exists.
    """
    ensure_csv_exists()
    names_first = ["Alex", "Jordan", "Priya", "Rahul", "Sam", "Taylor", "Morgan", "Ananya", "David", "Elena", "Rohan", "Aarav", "Sara", "Liam", "Zoe"]
    names_last = ["Sharma", "Patel", "Gehlot", "Verma", "Smith", "Johnson", "Williams", "Gupta", "Mehta", "Singh", "Kumar", "Iyer", "Rao", "Taylor"]
    genders = ["Male", "Female", "Other"]
    meal_regs = ["Regular", "Moderate", "Irregular"]
    smokings = ["No", "No", "No", "Occasional", "Regular"]
    alcohols = ["No", "No", "Occasional", "Occasional", "Regular"]
    caffeines = ["Low (0-1 cups)", "Moderate (2-3 cups)", "High (4+ cups)"]
    concerns_list = ["None", "Occasional fatigue", "High workload stress", "Back stiffness", "Poor sleep latency", "Mild screen eye strain", "None"]

    records = []
    base_id = 1000

    for i in range(1, n + 1):
        uid = f"WC{base_id + i}"
        name = f"{random.choice(names_first)} {random.choice(names_last)}"
        age = random.randint(18, 65)
        gender = random.choice(genders)
        height = random.randint(152, 190)
        weight = random.randint(48, 98)

        # Profile archetypes for diverse distribution
        archetype = random.choice(["high_fit", "balanced", "stressed_student", "sedentary_tech"])
        if archetype == "high_fit":
            steps = random.randint(8000, 15000)
            ex_dur = random.randint(35, 75)
            ex_freq = random.randint(4, 6)
            sleep = round(random.uniform(7.0, 8.5), 1)
            water = round(random.uniform(2.5, 4.0), 1)
            screen = round(random.uniform(3.0, 6.0), 1)
            work = round(random.uniform(6.0, 8.5), 1)
            stress = random.randint(1, 4)
            mood = random.randint(7, 10)
            energy = random.randint(7, 10)
            meal = "Regular"
            smk = "No"
            alc = random.choice(["No", "Occasional"])
            caf = random.choice(["Low (0-1 cups)", "Moderate (2-3 cups)"])
        elif archetype == "balanced":
            steps = random.randint(6000, 9500)
            ex_dur = random.randint(20, 45)
            ex_freq = random.randint(2, 4)
            sleep = round(random.uniform(6.5, 8.0), 1)
            water = round(random.uniform(2.0, 3.2), 1)
            screen = round(random.uniform(4.5, 7.5), 1)
            work = round(random.uniform(7.0, 9.0), 1)
            stress = random.randint(3, 6)
            mood = random.randint(6, 8)
            energy = random.randint(5, 8)
            meal = random.choice(["Regular", "Moderate"])
            smk = "No"
            alc = random.choice(["No", "Occasional"])
            caf = "Moderate (2-3 cups)"
        elif archetype == "stressed_student":
            steps = random.randint(3000, 6500)
            ex_dur = random.randint(0, 25)
            ex_freq = random.randint(0, 2)
            sleep = round(random.uniform(4.5, 6.5), 1)
            water = round(random.uniform(1.0, 2.2), 1)
            screen = round(random.uniform(7.5, 11.0), 1)
            work = round(random.uniform(9.0, 13.0), 1)
            stress = random.randint(6, 10)
            mood = random.randint(3, 6)
            energy = random.randint(3, 6)
            meal = random.choice(["Moderate", "Irregular"])
            smk = random.choice(smokings)
            alc = random.choice(alcohols)
            caf = random.choice(["Moderate (2-3 cups)", "High (4+ cups)"])
        else: # sedentary_tech
            steps = random.randint(2000, 5000)
            ex_dur = random.randint(0, 20)
            ex_freq = random.randint(0, 2)
            sleep = round(random.uniform(5.5, 7.0), 1)
            water = round(random.uniform(1.2, 2.0), 1)
            screen = round(random.uniform(8.5, 13.0), 1)
            work = round(random.uniform(8.0, 11.5), 1)
            stress = random.randint(5, 8)
            mood = random.randint(4, 7)
            energy = random.randint(3, 6)
            meal = random.choice(meal_regs)
            smk = random.choice(smokings)
            alc = random.choice(alcohols)
            caf = random.choice(caffeines)

        row_dict = {
            "User_ID": uid,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Height": height,
            "Weight": weight,
            "Steps": steps,
            "Exercise_Duration": ex_dur,
            "Exercise_Frequency": ex_freq,
            "Sleep_Hours": sleep,
            "Water_Intake": water,
            "Screen_Time": screen,
            "Work_Study_Hours": work,
            "Meal_Regularity": meal,
            "Stress_Level": stress,
            "Mood_Level": mood,
            "Energy_Level": energy,
            "Smoking": smk,
            "Alcohol": alc,
            "Caffeine": caf,
            "Concerns": random.choice(concerns_list),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        scoring = calculate_wellness_score(row_dict)
        row_dict["Wellness_Score"] = scoring["Wellness_Score"]
        row_dict["Wellness_Category"] = scoring["Wellness_Category"]
        records.append(row_dict)

    df_seed = pd.DataFrame(records)
    df_seed.to_csv(CSV_PATH, index=False)
    return len(records)
