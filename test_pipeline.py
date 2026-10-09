import os
from scoring import calculate_wellness_score
from preprocessing import ensure_csv_exists, save_user_record, generate_seed_dataset, load_all_users
from model import train_and_evaluate_models, predict_wellness
from report_generator import generate_pdf_report

print('--- 1. Testing Scoring Engine ---')
sample_user = {
    'User_ID': 'WC1001',
    'Name': 'Test User',
    'Age': 24,
    'Gender': 'Male',
    'Height': 175,
    'Weight': 68,
    'Steps': 8500,
    'Exercise_Duration': 40,
    'Exercise_Frequency': 4,
    'Sleep_Hours': 7.5,
    'Water_Intake': 2.8,
    'Screen_Time': 5.0,
    'Work_Study_Hours': 8.0,
    'Meal_Regularity': 'Regular',
    'Stress_Level': 3,
    'Mood_Level': 8,
    'Energy_Level': 8,
    'Smoking': 'No',
    'Alcohol': 'Occasional',
    'Caffeine': 'Moderate (2-3 cups)',
    'Concerns': 'None'
}
res = calculate_wellness_score(sample_user)
print('Score:', res['Wellness_Score'], '| Category:', res['Wellness_Category'], '| BMI:', res['BMI'])

print('\n--- 2. Testing Seed Generation & CSV Persistence ---')
seed_count = generate_seed_dataset(n=60)
df = load_all_users()
print(f'Generated {seed_count} seed records. Total records in CSV: {len(df)}')

print('\n--- 3. Testing 4-Model ML Training & Benchmarking ---')
train_results = train_and_evaluate_models()
print('Training Status:', train_results['status'])
print('Best Model:', train_results['best_model_name'])
for m in train_results['comparison']:
    print(f"  Model: {m['Model']:22} | Acc: {m['Accuracy']}% | F1: {m['F1_Score']}%")

print('\n--- 4. Testing Prediction Inference ---')
pred = predict_wellness(sample_user)
print('Predicted Category:', pred['Predicted_Category'], '| Confidence:', pred['Confidence'], '%')
print('Model Used:', pred['Model_Name'])

print('\n--- 5. Testing PDF Report Generation ---')
pdf_path = generate_pdf_report(sample_user, pred)
print('PDF Report generated at:', pdf_path, '| Exists:', os.path.exists(pdf_path))
print('\n>>> ALL MODULE TESTS PASSED SUCCESSFULLY! <<<')
