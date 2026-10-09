import unittest
import json
from app import app
from preprocessing import save_user_record, validate_user_input

class WellCareAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Ensure a test user exists
        _, _, clean_user = validate_user_input({
            'User_ID': 'WC9999',
            'Name': 'Flask Test User',
            'Age': '28',
            'Gender': 'Female',
            'Height': '165',
            'Weight': '58',
            'Steps': '9200',
            'Exercise_Duration': '45',
            'Exercise_Frequency': '4',
            'Sleep_Hours': '8.0',
            'Water_Intake': '3.0',
            'Screen_Time': '4.5',
            'Work_Study_Hours': '7.5',
            'Meal_Regularity': 'Regular',
            'Stress_Level': '2',
            'Mood_Level': '9',
            'Energy_Level': '8',
            'Smoking': 'No',
            'Alcohol': 'No',
            'Caffeine': 'Low (0-1 cups)',
            'Concerns': 'None'
        })
        save_user_record(clean_user)

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'WellCare', response.data)

    def test_assessment_route(self):
        response = self.app.get('/assessment')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Personal Wellness Assessment', response.data)

    def test_predict_form_submission(self):
        payload = {
            'User_ID': 'WC9999',
            'Name': 'Flask Test User',
            'Age': '28',
            'Gender': 'Female',
            'Height': '165',
            'Weight': '58',
            'Steps': '9200',
            'Exercise_Duration': '45',
            'Exercise_Frequency': '4',
            'Sleep_Hours': '8.0',
            'Water_Intake': '3.0',
            'Screen_Time': '4.5',
            'Work_Study_Hours': '7.5',
            'Meal_Regularity': 'Regular',
            'Stress_Level': '2',
            'Mood_Level': '9',
            'Energy_Level': '8',
            'Smoking': 'No',
            'Alcohol': 'No',
            'Caffeine': 'Low (0-1 cups)',
            'Concerns': 'None'
        }
        response = self.app.post('/predict', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Personal Wellness Intelligence Profile', response.data)
        self.assertIn(b'Flask Test User', response.data)

    def test_dashboard_route(self):
        response = self.app.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Machine Learning Model Benchmark', response.data)

    def test_history_route(self):
        response = self.app.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User History &amp; Assessments', response.data)

    def test_pdf_download_route(self):
        response = self.app.get('/report/WC9999')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/pdf')

    def test_api_users(self):
        response = self.app.get('/api/users')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertGreaterEqual(data['count'], 1)

    def test_api_dashboard(self):
        response = self.app.get('/api/dashboard')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('category_distribution', data)

    def test_api_predict(self):
        payload = {
            'User_ID': 'WC8888',
            'Name': 'API Client User',
            'Age': 32,
            'Gender': 'Male',
            'Height': 180,
            'Weight': 75,
            'Steps': 7500,
            'Exercise_Duration': 30,
            'Exercise_Frequency': 3,
            'Sleep_Hours': 7.0,
            'Water_Intake': 2.5,
            'Screen_Time': 6.0,
            'Work_Study_Hours': 8.0,
            'Meal_Regularity': 'Moderate',
            'Stress_Level': 5,
            'Mood_Level': 6,
            'Energy_Level': 6,
            'Smoking': 'No',
            'Alcohol': 'Occasional',
            'Caffeine': 'Moderate (2-3 cups)',
            'Concerns': 'Occasional fatigue'
        }
        response = self.app.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('wellness_score', data)
        self.assertIn('category', data)
        self.assertIn('recommendations', data)

if __name__ == '__main__':
    unittest.main()
