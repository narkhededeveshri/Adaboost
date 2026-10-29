import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = 'adaboost.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML Template with embedded modern responsive CSS & layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Model Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border-color: #d1d5db;
            --radius: 12px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            padding: 40px;
            margin-bottom: 30px;
        }

        header {
            text-align: center;
            margin-bottom: 35px;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 10px;
        }

        header p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        .alert-error {
            background-color: #fee2e2;
            border: 1px solid #f87171;
            color: #991b1b;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            text-align: center;
            font-weight: 500;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-main);
        }

        input, select {
            padding: 12px 14px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
            background: #f9fafb;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .full-width {
            grid-column: 1 / -1;
            margin-top: 10px;
        }

        button {
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            width: 100%;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-card {
            width: 100%;
            max-width: 950px;
            background: #ffffff;
            border-radius: var(--radius);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            padding: 30px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
            margin-bottom: 20px;
        }

        .result-card h2 {
            font-size: 1.5rem;
            margin-bottom: 10px;
            color: var(--text-main);
        }

        .prediction-badge {
            display: inline-block;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            background: #eef2ff;
            padding: 12px 35px;
            border-radius: 50px;
            margin-top: 10px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AdaBoost Prediction Portal</h1>
            <p>Enter candidate or student details below to generate real-time classification results.</p>
        </header>

        {% if error %}
        <div class="alert-error">
            {{ error }}
        </div>
        {% endif %}

        <form method="POST" action="/">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" step="1" required value="{{ form_data.get('age', '') }}">
            </div>

            <div class="form-group">
                <label for="gender">Gender (Encoded Code)</label>
                <input type="number" id="gender" name="gender" step="1" required placeholder="e.g., 0 or 1" value="{{ form_data.get('gender', '') }}">
            </div>

            <div class="form-group">
                <label for="state">State (Encoded Code)</label>
                <input type="number" id="state" name="state" step="1" required placeholder="e.g., State code" value="{{ form_data.get('state', '') }}">
            </div>

            <div class="form-group">
                <label for="family_income">Family Income</label>
                <input type="number" id="family_income" name="family_income" step="any" required value="{{ form_data.get('family_income', '') }}">
            </div>

            <div class="form-group">
                <label for="high_school_gpa">High School GPA</label>
                <input type="number" id="high_school_gpa" name="high_school_gpa" step="0.01" required value="{{ form_data.get('high_school_gpa', '') }}">
            </div>

            <div class="form-group">
                <label for="sat_score">SAT Score</label>
                <input type="number" id="sat_score" name="sat_score" step="1" required value="{{ form_data.get('sat_score', '') }}">
            </div>

            <div class="form-group">
                <label for="act_score">ACT Score</label>
                <input type="number" id="act_score" name="act_score" step="1" required value="{{ form_data.get('act_score', '') }}">
            </div>

            <div class="form-group">
                <label for="attendance_rate">Attendance Rate</label>
                <input type="number" id="attendance_rate" name="attendance_rate" step="0.01" required value="{{ form_data.get('attendance_rate', '') }}">
            </div>

            <div class="form-group">
                <label for="ap_courses">AP Courses</label>
                <input type="number" id="ap_courses" name="ap_courses" step="1" required value="{{ form_data.get('ap_courses', '') }}">
            </div>

            <div class="form-group">
                <label for="extracurricular_count">Extracurricular Count</label>
                <input type="number" id="extracurricular_count" name="extracurricular_count" step="1" required value="{{ form_data.get('extracurricular_count', '') }}">
            </div>

            <div class="form-group">
                <label for="volunteer_hours">Volunteer Hours</label>
                <input type="number" id="volunteer_hours" name="volunteer_hours" step="1" required value="{{ form_data.get('volunteer_hours', '') }}">
            </div>

            <div class="form-group">
                <label for="leadership_positions">Leadership Positions</label>
                <input type="number" id="leadership_positions" name="leadership_positions" step="1" required value="{{ form_data.get('leadership_positions', '') }}">
            </div>

            <div class="form-group">
                <label for="coding_projects">Coding Projects</label>
                <input type="number" id="coding_projects" name="coding_projects" step="1" required value="{{ form_data.get('coding_projects', '') }}">
            </div>

            <div class="form-group">
                <label for="social_media_hours">Social Media Hours</label>
                <input type="number" id="social_media_hours" name="social_media_hours" step="0.1" required value="{{ form_data.get('social_media_hours', '') }}">
            </div>

            <div class="form-group">
                <label for="online_certifications">Online Certifications</label>
                <input type="number" id="online_certifications" name="online_certifications" step="1" required value="{{ form_data.get('online_certifications', '') }}">
            </div>

            <div class="form-group">
                <label for="essay_score">Essay Score</label>
                <input type="number" id="essay_score" name="essay_score" step="0.1" required value="{{ form_data.get('essay_score', '') }}">
            </div>

            <div class="form-group">
                <label for="recommendation_score">Recommendation Score</label>
                <input type="number" id="recommendation_score" name="recommendation_score" step="0.1" required value="{{ form_data.get('recommendation_score', '') }}">
            </div>

            <div class="form-group">
                <label for="interview_score">Interview Score</label>
                <input type="number" id="interview_score" name="interview_score" step="0.1" required value="{{ form_data.get('interview_score', '') }}">
            </div>

            <div class="form-group full-width">
                <button type="submit">Run Prediction</button>
            </div>
        </form>
    </div>

    {% if prediction is not none %}
    <div class="result-card">
        <h2>Prediction Result</h2>
        <p>The AdaBoost model classification outcome is:</p>
        <div class="prediction-badge">{{ prediction }}</div>
    </div>
    {% endif %}

    <footer>
        <p>Deployed with Flask &bull; Powered by scikit-learn AdaBoost</p>
    </footer>
</body>
</html>
"""

# Exact feature order extracted from your adaboost.pkl file
FEATURE_NAMES = [
    'age', 'gender', 'state', 'family_income', 'high_school_gpa', 
    'sat_score', 'act_score', 'attendance_rate', 'ap_courses', 
    'extracurricular_count', 'volunteer_hours', 'leadership_positions', 
    'coding_projects', 'social_media_hours', 'online_certifications', 
    'essay_score', 'recommendation_score', 'interview_score'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    form_data = {}

    if request.method == 'POST':
        if model is None:
            error = "Model file 'adaboost.pkl' not found or failed to load."
        else:
            try:
                form_data = request.form
                features = []
                for feat in FEATURE_NAMES:
                    val = form_data.get(feat)
                    if val is None or val.strip() == '':
                        raise ValueError(f"Missing value for {feat}")
                    features.append(float(val))
                
                features_array = np.array([features])
                pred = model.predict(features_array)[0]
                prediction = str(pred)
            except Exception as e:
                error = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error, form_data=form_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
