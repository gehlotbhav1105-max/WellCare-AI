"""
WellCare AI - Recommendation Engine
===================================
Rule-based recommendation system delivering actionable, multi-faceted wellness
guidance based on user lifestyle metrics, scoring breakdowns, and identified risks.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

from typing import Dict, Any, List


def generate_recommendations(user_data: Dict[str, Any], scoring_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates structured, rule-based recommendations tailored to the user's specific inputs.
    
    Returns:
        Dict containing:
        - priority_actions: Top urgent/high-impact improvements
        - general_advice: Category-wise advice list
        - strengths_affirmations: Positive reinforcement for well-performing areas
        - disclaimer: Mandatory educational safety disclaimer
    """
    priority_actions: List[Dict[str, str]] = []
    general_advice: List[Dict[str, str]] = []
    strengths_affirmations: List[str] = []

    # 1. Sleep Recommendations
    sleep = float(user_data.get("Sleep_Hours", 7) or 7)
    if sleep < 6.0:
        priority_actions.append({
            "category": "Sleep & Recovery",
            "icon": "fa-bed",
            "title": "Prioritize Sleep Duration (Currently < 6h)",
            "action": "Aim for at least 7–8 hours of restful sleep tonight. Set a soothing wind-down alarm 45 minutes before bedtime and turn off bright screens."
        })
    elif sleep < 7.0:
        general_advice.append({
            "category": "Sleep & Recovery",
            "icon": "fa-moon",
            "title": "Enhance Sleep Consistency",
            "action": "You're close to optimal sleep (6–7h). Aiming for an additional 30–60 minutes can significantly improve daytime focus and metabolic recovery."
        })
    elif sleep > 9.5:
        general_advice.append({
            "category": "Sleep & Recovery",
            "icon": "fa-clock",
            "title": "Optimize Sleep Timing",
            "action": "Sleeping more than 9.5 hours regularly may indicate sluggish circadian rhythm. Consider standardizing fixed wake-up and sleep times."
        })
    else:
        strengths_affirmations.append("Excellent sleep duration (7–9 hours) supporting cellular restoration and neurocognitive health.")

    # 2. Hydration Recommendations
    water = float(user_data.get("Water_Intake", 2.0) or 2.0)
    if water < 1.5:
        priority_actions.append({
            "category": "Hydration",
            "icon": "fa-tint",
            "title": "Boost Daily Fluid Intake (Currently < 1.5L)",
            "action": "Keep a reusable 1-liter water bottle at your desk and set hourly hydration reminders to reach 2.5–3.0 liters per day."
        })
    elif water < 2.5:
        general_advice.append({
            "category": "Hydration",
            "icon": "fa-tint",
            "title": "Maintain Steady Hydration",
            "action": "Increase fluid intake slightly by sipping herbal teas or infused water during afternoon work sessions."
        })
    else:
        strengths_affirmations.append("Healthy daily hydration levels maintaining optimal electrolyte balance and alertness.")

    # 3. Physical Activity & Daily Steps
    steps = float(user_data.get("Steps", 0) or 0)
    exercise_dur = float(user_data.get("Exercise_Duration", 0) or 0)
    exercise_freq = float(user_data.get("Exercise_Frequency", 0) or 0)

    if steps < 5000 and exercise_dur < 20:
        priority_actions.append({
            "category": "Physical Activity",
            "icon": "fa-walking",
            "title": "Increase Daily Movement",
            "action": "Incorporate a 15-minute brisk walk after meals and take phone calls while pacing to steadily build up to 7,500+ steps."
        })
    elif steps < 8000 or exercise_freq < 3:
        general_advice.append({
            "category": "Physical Activity",
            "icon": "fa-running",
            "title": "Establish Structured Workouts",
            "action": "Schedule 3 dedicated 30-minute moderate aerobic or resistance exercise sessions per week."
        })
    else:
        strengths_affirmations.append("Commendable physical activity and step volume promoting cardiovascular vigor.")

    # 4. Mental & Emotional Wellness (Stress / Mood / Energy)
    stress = float(user_data.get("Stress_Level", 5) or 5)
    mood = float(user_data.get("Mood_Level", 5) or 5)
    energy = float(user_data.get("Energy_Level", 5) or 5)

    if stress >= 7:
        priority_actions.append({
            "category": "Stress Management",
            "icon": "fa-heartbeat",
            "title": "Active Stress Decompression",
            "action": "Practice 5 minutes of box breathing (4s inhale, 4s hold, 4s exhale, 4s hold) twice daily. Consider delegating non-urgent tasks and speaking with a supportive mentor or peer."
        })
    elif stress >= 5:
        general_advice.append({
            "category": "Stress Management",
            "icon": "fa-spa",
            "title": "Micro-Break Integration",
            "action": "Take short 3-minute mental decompression breaks every 90 minutes of intensive cognitive work."
        })
    else:
        strengths_affirmations.append("Effective stress resilience and balanced mental equilibrium.")

    if mood <= 4 or energy <= 4:
        general_advice.append({
            "category": "Mood & Energy",
            "icon": "fa-sun",
            "title": "Vitality Boost via Morning Sun & Social Connection",
            "action": "Get 10–15 minutes of direct morning sunlight to elevate natural dopamine and cortisol rhythm, and schedule downtime with friends."
        })

    # 5. Screen Time & Digital Wellness
    screen = float(user_data.get("Screen_Time", 5.0) or 5.0)
    if screen >= 8.0:
        priority_actions.append({
            "category": "Digital Wellness",
            "icon": "fa-laptop",
            "title": "Mitigate High Screen Time (8+ hours/day)",
            "action": "Implement the 20-20-20 rule: Every 20 minutes, look at an object 20 feet away for 20 seconds. Use blue-light filters in the evening."
        })
    elif screen >= 6.0:
        general_advice.append({
            "category": "Digital Wellness",
            "icon": "fa-mobile-alt",
            "title": "Introduce Digital Sunset",
            "action": "Try placing mobile devices in another room 30 minutes before sleep to prevent digital over-stimulation."
        })
    else:
        strengths_affirmations.append("Well-managed screen exposure reducing eye strain and cognitive fatigue.")

    # 6. Work/Study Hours
    work_hrs = float(user_data.get("Work_Study_Hours", 8.0) or 8.0)
    if work_hrs >= 10.0:
        general_advice.append({
            "category": "Work-Life Harmony",
            "icon": "fa-briefcase",
            "title": "Boundary Management",
            "action": "Long working hours (>10h) increase burnout risk. Define a clear end-of-work ritual to transition into restorative personal time."
        })

    # 7. Diet & Lifestyle Habits (Meal Regularity, Smoking, Alcohol, Caffeine)
    meal_reg = str(user_data.get("Meal_Regularity", "Regular")).strip().lower()
    smoking = str(user_data.get("Smoking", "No")).strip().lower()
    alcohol = str(user_data.get("Alcohol", "No")).strip().lower()
    caffeine = str(user_data.get("Caffeine", "Low")).strip().lower()

    if "irreg" in meal_reg:
        priority_actions.append({
            "category": "Nutritional Rhythm",
            "icon": "fa-utensils",
            "title": "Standardize Meal Timing",
            "action": "Establish consistent windows for breakfast, lunch, and dinner to stabilize blood glucose and metabolic hormone levels."
        })

    if smoking not in ["no", "never"]:
        general_advice.append({
            "category": "Habit Moderation",
            "icon": "fa-smoking-ban",
            "title": "Tobacco Cessation Support",
            "action": "Reducing smoking frequency provides immediate benefits for lung vital capacity, arterial flexibility, and restorative sleep depth."
        })

    if alcohol in ["regular", "high", "daily"]:
        general_advice.append({
            "category": "Habit Moderation",
            "icon": "fa-cocktail",
            "title": "Alcohol Moderation",
            "action": "Alcohol disrupts REM sleep architecture and dehydrates the body. Try designating alcohol-free recovery days during the week."
        })

    if "high" in caffeine or "4+" in caffeine:
        general_advice.append({
            "category": "Habit Moderation",
            "icon": "fa-coffee",
            "title": "Caffeine Curfew",
            "action": "Cap daily coffee/energy drink intake to 1–2 cups and avoid caffeine intake after 2:00 PM to protect deep restorative sleep."
        })

    # Fallbacks if list is brief
    if not priority_actions and general_advice:
        priority_actions.append(general_advice.pop(0))

    if not priority_actions and not general_advice:
        general_advice.append({
            "category": "General Maintenance",
            "icon": "fa-check-circle",
            "title": "Maintain Peak Wellness Routine",
            "action": "Your lifestyle profile is well-balanced. Keep tracking your metrics weekly to sustain long-term energy and vitality."
        })

    if not strengths_affirmations:
        strengths_affirmations.append("Proactive engagement in personal health tracking and lifestyle optimization.")

    return {
        "priority_actions": priority_actions,
        "general_advice": general_advice,
        "strengths_affirmations": strengths_affirmations,
        "disclaimer": (
            "WellCare AI is an educational wellness-support system. Its results are "
            "general wellness indicators and should not be considered a medical "
            "diagnosis or a substitute for professional healthcare advice."
        )
    }
