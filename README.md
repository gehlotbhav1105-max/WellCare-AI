# WellCare AI – Intelligent Personal Wellness Monitoring & Prediction System
**Academic B.Tech AI/ML & Data Science Final-Year Project**

WellCare AI is an end-to-end intelligent personal wellness management system designed to monitor lifestyle health markers, compute multi-pillar wellness indices (0–100), and autonomously train, evaluate, and deploy Machine Learning classifiers (*Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting*).

---

## 🌟 Key Features

1. **Zero-Dataset Bootstrapping**:
   - Requires **no pre-existing dataset**.
   - Dynamically creates and updates `data/wellcare_users.csv` on user submissions.
   - Includes one-click **Demonstration Data Generation (60 records)** for instant academic evaluation.

2. **Transparent Multi-Pillar Scoring (0–100)**:
   - **Physical Fitness (25%)**: Steps, exercise duration, workout frequency, and BMI index.
   - **Sleep & Circadian Restoration (20%)**: Nightly sleep duration.
   - **Lifestyle & Hydration (20%)**: Fluid intake, screen time exposure, and work-life balance.
   - **Mental & Emotional Wellness (20%)**: Stress index, mood rating, and daytime energy.
   - **Diet & Habit Moderation (15%)**: Meal regularity, caffeine, smoking, and alcohol habits.

3. **4-Algorithm Machine Learning Suite**:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier (100 Trees)
   - Gradient Boosting Classifier
   - Benchmark evaluation across **Accuracy, Precision, Recall, F1-Score, and Confusion Matrices**.
   - Automatic serialization of the winning pipeline to `models/wellcare_model.pkl`.

4. **Explainable AI & Recommendations**:
   - Class confidence distributions and feature importance rankings.
   - Tailored rule-based lifestyle suggestions (Sleep, Hydration, Steps, Stress Decompression).

5. **Clinical-Grade ReportLab PDF Generation**:
   - Downloadable vector PDF summary with demographic cards, pillar tables, and customized action items.

6. **Interactive Population Analytics Dashboard**:
   - Multi-user KPIs, category distributions, lifestyle factor averages, and real-time model retraining.

---

## 📁 Project Directory Structure

```text
WellCare_AI/
│
├── data/
│   └── wellcare_users.csv          # Persistent multi-user dataset
│
├── models/
│   └── wellcare_model.pkl          # Serialized winner ML pipeline & metadata
│
├── notebooks/
│   └── WellCare_AI.ipynb           # Complete 25-section sequential Jupyter Notebook
│
├── templates/
│   ├── base.html                   # Master layout with modern navbar & footer
│   ├── index.html                  # Landing Home page with hero & workflow
│   ├── assessment.html             # Multi-section interactive questionnaire
│   ├── result.html                 # AI prediction profile & Chart.js radar charts
│   ├── dashboard.html              # Population analytics & ML benchmark table
│   └── history.html                # Multi-user history table with search/filter
│
├── static/
│   ├── css/
│   │   └── style.css               # Modern healthcare design system
│   └── js/
│       └── script.js               # Dynamic sliders, BMI calc, and form validation
│
├── reports/
│   └── WellCare_Report_*.pdf       # Generated ReportLab PDF reports
│
├── app.py                          # Flask web application & REST API routes
├── preprocessing.py                # Input validation, CSV I/O, & ColumnTransformer
├── scoring.py                      # Transparent 5-pillar wellness scoring engine
├── model.py                        # 4-Model ML training, benchmarking, & inference
├── recommendations.py              # Rule-based personalized wellness guidance
├── report_generator.py             # Vector PDF report compiler (ReportLab)
├── requirements.txt                # Python dependencies
├── PROJECT_DOCUMENTATION.md        # Complete 29-section academic project report
└── README.md                       # Setup and operational guide
```

---

## 🚀 Quickstart Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Git (optional) and VS Code / Jupyter Lab.

### 2. Setup Virtual Environment

```bash
# Navigate to project directory
cd WellCare_AI

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web Application

```bash
python app.py
```

Open your browser and navigate to:
👉 **`http://localhost:5000`** (or `http://127.0.0.1:5000`)

---

## 📓 Running the 25-Section Jupyter Notebook

To explore or demonstrate the step-by-step data science lifecycle:

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/WellCare_AI.ipynb
```

The notebook contains 25 clearly delineated sections with code, explanations, and visualizations:
- **Sections 1–6**: Imports, storage initialization, form input, validation, and CSV persistence.
- **Sections 7–11**: Data loading, cleaning, feature engineering, scoring, and categories.
- **Sections 12–14**: EDA visual plots, ColumnTransformer pipelines, and stratified split.
- **Sections 15–21**: 4-Model training, evaluation, comparison, selection, and serialization.
- **Sections 22–25**: Live prediction, recommendations, radar dashboards, and PDF compilation.

---

## 🌐 REST API Documentation

WellCare AI exposes JSON REST APIs for integration:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/users` | Retrieve all registered users in JSON format |
| `GET` | `/api/user/<user_id>` | Fetch user profile, score breakdown, and prediction |
| `POST` | `/api/predict` | Submit user metrics JSON payload and receive prediction |
| `GET` | `/api/dashboard` | Get population averages and ML benchmark metrics |
| `POST` | `/api/retrain` | Trigger retraining of all 4 ML classifiers |

### Sample `POST /api/predict` Request:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "User_ID": "WC1001",
    "Name": "Alex Morgan",
    "Age": 26,
    "Gender": "Male",
    "Height": 178,
    "Weight": 72,
    "Steps": 9500,
    "Exercise_Duration": 45,
    "Exercise_Frequency": 4,
    "Sleep_Hours": 7.5,
    "Water_Intake": 3.0,
    "Screen_Time": 5.0,
    "Work_Study_Hours": 8.0,
    "Meal_Regularity": "Regular",
    "Stress_Level": 3,
    "Mood_Level": 8,
    "Energy_Level": 8,
    "Smoking": "No",
    "Alcohol": "No",
    "Caffeine": "Moderate (2-3 cups)",
    "Concerns": "None"
  }'
```

---

## 🧪 Automated Testing

Execute the automated unit and integration test suite:

```bash
# Run modular pipeline verification
python test_pipeline.py

# Run Flask web route and API tests
python test_app.py
```

---

## ⚠️ Non-Medical Educational Disclaimer

> **IMPORTANT**: WellCare AI is an educational wellness-support system developed for lifestyle awareness and academic demonstration. Its results are general wellness indicators and should **not** be considered a medical diagnosis, clinical treatment, or substitute for professional healthcare advice.
