"""
WellCare AI - Scoring Engine
============================
Provides transparent, multi-dimensional lifestyle & wellness scoring (0-100).
Calculates category classifications, individual factor ratings, and key strength/improvement indicators.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

from typing import Dict, Any, Tuple, List


def calculate_bmi(height_cm: float, weight_kg: float) -> Tuple[float, str]:
    """Calculates BMI and returns BMI value and category."""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0, "Unknown"
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"
    return bmi, category


def calculate_wellness_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes a transparent 0-100 wellness score based on 5 core pillars:
    1. Physical Wellness (max 25 pts)
    2. Sleep & Recovery (max 20 pts)
    3. Lifestyle & Hydration (max 20 pts)
    4. Mental & Emotional Wellness (max 20 pts)
    5. Diet & Habit Factors (max 15 pts)

    Returns comprehensive score breakdown, sub-scores, category, and factor levels.
    """
    # 1. Physical Wellness (25 pts max)
    steps = float(data.get("Steps", 0) or 0)
    exercise_dur = float(data.get("Exercise_Duration", 0) or 0)
    exercise_freq = float(data.get("Exercise_Frequency", 0) or 0)
    height = float(data.get("Height", 170) or 170)
    weight = float(data.get("Weight", 65) or 65)
    bmi, bmi_cat = calculate_bmi(height, weight)

    # Steps points (max 10)
    if steps >= 10000:
        steps_pts = 10.0
    elif steps >= 7500:
        steps_pts = 8.0
    elif steps >= 5000:
        steps_pts = 6.0
    elif steps >= 3000:
        steps_pts = 4.0
    else:
        steps_pts = 2.0

    # Exercise Duration points (max 8)
    if exercise_dur >= 45:
        dur_pts = 8.0
    elif exercise_dur >= 30:
        dur_pts = 6.0
    elif exercise_dur >= 15:
        dur_pts = 4.0
    elif exercise_dur > 0:
        dur_pts = 2.0
    else:
        dur_pts = 0.0

    # Exercise Frequency points (max 7)
    if exercise_freq >= 5:
        freq_pts = 7.0
    elif exercise_freq >= 3:
        freq_pts = 5.0
    elif exercise_freq >= 1:
        freq_pts = 3.0
    else:
        freq_pts = 0.0

    # BMI adjustment
    bmi_adj = 0.0
    if bmi_cat in ["Overweight", "Underweight"]:
        bmi_adj = -1.0
    elif bmi_cat == "Obese":
        bmi_adj = -2.0

    physical_score = max(0.0, min(25.0, steps_pts + dur_pts + freq_pts + bmi_adj))

    # 2. Sleep & Recovery (20 pts max)
    sleep_hours = float(data.get("Sleep_Hours", 7) or 7)
    if 7.0 <= sleep_hours <= 9.0:
        sleep_score = 20.0
    elif (6.0 <= sleep_hours < 7.0) or (9.0 < sleep_hours <= 10.0):
        sleep_score = 15.0
    elif 5.0 <= sleep_hours < 6.0:
        sleep_score = 10.0
    else:
        sleep_score = 5.0

    # 3. Lifestyle & Hydration (20 pts max)
    water_intake = float(data.get("Water_Intake", 2.0) or 2.0)
    screen_time = float(data.get("Screen_Time", 5.0) or 5.0)
    work_hours = float(data.get("Work_Study_Hours", 8.0) or 8.0)

    # Water (max 8 pts)
    if water_intake >= 3.0:
        water_pts = 8.0
    elif water_intake >= 2.0:
        water_pts = 6.0
    elif water_intake >= 1.0:
        water_pts = 4.0
    else:
        water_pts = 1.0

    # Screen time (max 6 pts)
    if screen_time <= 3.0:
        screen_pts = 6.0
    elif screen_time <= 6.0:
        screen_pts = 4.0
    elif screen_time <= 8.0:
        screen_pts = 2.0
    else:
        screen_pts = 0.0

    # Work/Study hours (max 6 pts)
    if work_hours <= 8.0:
        work_pts = 6.0
    elif work_hours <= 10.0:
        work_pts = 4.0
    elif work_hours <= 12.0:
        work_pts = 2.0
    else:
        work_pts = 0.0

    lifestyle_score = max(0.0, min(20.0, water_pts + screen_pts + work_pts))

    # 4. Mental & Emotional Wellness (20 pts max)
    stress = float(data.get("Stress_Level", 5) or 5)
    mood = float(data.get("Mood_Level", 5) or 5)
    energy = float(data.get("Energy_Level", 5) or 5)

    # Stress (max 8 pts, inverted)
    if stress <= 2:
        stress_pts = 8.0
    elif stress <= 4:
        stress_pts = 6.0
    elif stress <= 6:
        stress_pts = 4.0
    elif stress <= 8:
        stress_pts = 2.0
    else:
        stress_pts = 0.0

    # Mood (max 6 pts)
    if mood >= 8:
        mood_pts = 6.0
    elif mood >= 6:
        mood_pts = 4.5
    elif mood >= 4:
        mood_pts = 3.0
    else:
        mood_pts = 1.0

    # Energy (max 6 pts)
    if energy >= 8:
        energy_pts = 6.0
    elif energy >= 6:
        energy_pts = 4.5
    elif energy >= 4:
        energy_pts = 3.0
    else:
        energy_pts = 1.0

    mental_score = max(0.0, min(20.0, stress_pts + mood_pts + energy_pts))

    # 5. Diet & Habit Factors (15 pts max)
    meal_reg = str(data.get("Meal_Regularity", "Regular")).strip().capitalize()
    smoking = str(data.get("Smoking", "No")).strip().capitalize()
    alcohol = str(data.get("Alcohol", "No")).strip().capitalize()
    caffeine = str(data.get("Caffeine", "Low")).strip()

    # Meal Regularity (max 5 pts)
    if "reg" in meal_reg.lower() and "irreg" not in meal_reg.lower():
        meal_pts = 5.0
    elif "mod" in meal_reg.lower():
        meal_pts = 3.0
    else:
        meal_pts = 1.0

    # Smoking (max 4 pts)
    if smoking.lower() in ["no", "never"]:
        smoking_pts = 4.0
    elif "occ" in smoking.lower():
        smoking_pts = 2.0
    else:
        smoking_pts = 0.0

    # Alcohol (max 3 pts)
    if alcohol.lower() in ["no", "never"]:
        alcohol_pts = 3.0
    elif "occ" in alcohol.lower():
        alcohol_pts = 2.0
    else:
        alcohol_pts = 0.0

    # Caffeine (max 3 pts)
    if "low" in caffeine.lower() or "0-1" in caffeine.lower():
        caffeine_pts = 3.0
    elif "mod" in caffeine.lower() or "2-3" in caffeine.lower():
        caffeine_pts = 2.0
    else:
        caffeine_pts = 0.5

    habit_score = max(0.0, min(15.0, meal_pts + smoking_pts + alcohol_pts + caffeine_pts))

    # Total Wellness Score (0 - 100)
    total_score = round(physical_score + sleep_score + lifestyle_score + mental_score + habit_score, 1)
    total_score = max(0.0, min(100.0, total_score))

    # Category Mapping
    category = get_wellness_category(total_score)

    # Granular Factor Ratings
    factor_ratings = {
        "Sleep": "Optimal" if sleep_score >= 15 else ("Moderate" if sleep_score >= 10 else "Needs Attention"),
        "Physical Activity": "Optimal" if physical_score >= 18 else ("Moderate" if physical_score >= 12 else "Needs Attention"),
        "Hydration": "Optimal" if water_pts >= 6 else ("Moderate" if water_pts >= 4 else "Needs Attention"),
        "Stress Management": "Optimal" if stress <= 3 else ("Moderate" if stress <= 6 else "Needs Attention"),
        "Screen Time": "Optimal" if screen_time <= 4 else ("Moderate" if screen_time <= 7 else "Needs Attention"),
        "Emotional Well-being": "Optimal" if (mood >= 7 and energy >= 7) else ("Moderate" if (mood >= 5 and energy >= 5) else "Needs Attention"),
        "Habits & Nutrition": "Optimal" if habit_score >= 11 else ("Moderate" if habit_score >= 7 else "Needs Attention"),
    }

    # Identify Key Strengths & Areas to Improve
    strengths: List[str] = []
    improvements: List[str] = []

    for factor, status in factor_ratings.items():
        if status == "Optimal":
            strengths.append(factor)
        elif status == "Needs Attention":
            improvements.append(factor)

    if not strengths:
        strengths.append("General Health Awareness")
    if not improvements:
        improvements.append("Maintain Current Healthy Rhythm")

    return {
        "Wellness_Score": total_score,
        "Wellness_Category": category,
        "BMI": bmi,
        "BMI_Category": bmi_cat,
        "Pillars": {
            "Physical": round((physical_score / 25.0) * 100, 1),
            "Sleep": round((sleep_score / 20.0) * 100, 1),
            "Lifestyle": round((lifestyle_score / 20.0) * 100, 1),
            "Mental": round((mental_score / 20.0) * 100, 1),
            "Habits": round((habit_score / 15.0) * 100, 1),
        },
        "Sub_Scores": {
            "Physical_Score": physical_score,
            "Sleep_Score": sleep_score,
            "Lifestyle_Score": lifestyle_score,
            "Mental_Score": mental_score,
            "Habit_Score": habit_score,
        },
        "Factor_Ratings": factor_ratings,
        "Strengths": strengths,
        "Improvements": improvements,
    }


def get_wellness_category(score: float) -> str:
    """Classifies wellness score into designated categories."""
    if score >= 85.0:
        return "Excellent Wellness"
    elif score >= 70.0:
        return "Good Wellness"
    elif score >= 50.0:
        return "Moderate Wellness"
    else:
        return "Needs Attention"
