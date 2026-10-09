"""
WellCare AI - Flask Web Application & REST API
==============================================
Main server providing web routes, assessment workflows, predictive inference,
interactive analytics dashboards, user history management, PDF report downloads,
and REST API endpoints.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

import os
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
import pandas as pd
import numpy as np

from model import (
    load_model_bundle,
    predict_wellness,
    train_and_evaluate_models,
    MIN_RECORDS_FOR_ML,
)
from preprocessing import (
    CSV_PATH,
    ensure_csv_exists,
    generate_next_user_id,
    generate_seed_dataset,
    get_user_by_id,
    load_all_users,
    save_user_record,
    validate_user_input,
)
from recommendations import generate_recommendations
from report_generator import generate_pdf_report
from scoring import calculate_wellness_score

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "wellcare-ai-secret-key-2026")

# Initialize storage on startup
ensure_csv_exists()


@app.route("/")
def index():
    """Landing Home Page with overview and quick metrics."""
    df = load_all_users()
    total_users = len(df)
    avg_score = round(float(df["Wellness_Score"].mean()), 1) if not df.empty and "Wellness_Score" in df.columns and pd.notna(df["Wellness_Score"].mean()) else 0.0
    model_bundle = load_model_bundle()
    best_model = model_bundle.get("best_model_name", "Pending Training") if model_bundle else "Pending Training"
    
    return render_template(
        "index.html",
        total_users=total_users,
        avg_score=avg_score,
        best_model=best_model,
    )


@app.route("/assessment")
def assessment():
    """Wellness Assessment Form page."""
    next_id = generate_next_user_id()
    return render_template("assessment.html", next_user_id=next_id)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles assessment form submission, saves record, and redirects to result."""
    form_data = request.form.to_dict()
    is_valid, err_msg, clean_data = validate_user_input(form_data)

    if not is_valid:
        flash(f"Input Error: {err_msg}", "danger")
        return redirect(url_for("assessment"))

    # Save to CSV
    success, save_msg = save_user_record(clean_data)
    if not success:
        flash(f"Storage Error: {save_msg}", "danger")
        return redirect(url_for("assessment"))

    user_id = clean_data["User_ID"]
    flash(f"Assessment completed successfully for User {user_id}!", "success")
    return redirect(url_for("view_result", user_id=user_id))


@app.route("/result/<user_id>")
def view_result(user_id):
    """Displays detailed interactive prediction result for a specific user."""
    user_data = get_user_by_id(user_id)
    if not user_data:
        flash(f"User with ID '{user_id}' not found.", "warning")
        return redirect(url_for("history"))

    # Run prediction & scoring
    prediction = predict_wellness(user_data)
    scoring = prediction["Scoring_Details"]
    recommendations = generate_recommendations(user_data, scoring)

    return render_template(
        "result.html",
        user=user_data,
        prediction=prediction,
        scoring=scoring,
        recommendations=recommendations,
    )


@app.route("/dashboard")
def dashboard():
    """Aggregate population analytics dashboard and ML benchmark comparison."""
    df = load_all_users()
    total_users = len(df)

    if total_users > 0:
        avg_score = round(float(df["Wellness_Score"].mean()), 1) if "Wellness_Score" in df.columns else 0.0
        max_score = round(float(df["Wellness_Score"].max()), 1) if "Wellness_Score" in df.columns else 0.0
        min_score = round(float(df["Wellness_Score"].min()), 1) if "Wellness_Score" in df.columns else 0.0
        
        # Category distribution
        cat_counts = df["Wellness_Category"].value_counts().to_dict() if "Wellness_Category" in df.columns else {}
        
        # Averages for lifestyle features
        avg_sleep = round(float(df["Sleep_Hours"].mean()), 1) if "Sleep_Hours" in df.columns else 0.0
        avg_steps = int(df["Steps"].mean()) if "Steps" in df.columns else 0
        avg_stress = round(float(df["Stress_Level"].mean()), 1) if "Stress_Level" in df.columns else 0.0
        avg_water = round(float(df["Water_Intake"].mean()), 1) if "Water_Intake" in df.columns else 0.0
        avg_screen = round(float(df["Screen_Time"].mean()), 1) if "Screen_Time" in df.columns else 0.0
    else:
        avg_score = max_score = min_score = avg_sleep = avg_steps = avg_stress = avg_water = avg_screen = 0
        cat_counts = {}

    model_bundle = load_model_bundle()
    comparison_table = model_bundle.get("comparison", []) if model_bundle else []
    best_model_name = model_bundle.get("best_model_name", None) if model_bundle else None
    trained_at = model_bundle.get("trained_at", "Never") if model_bundle else "Never"
    feature_importances = model_bundle.get("feature_importances", {}) if model_bundle else {}

    return render_template(
        "dashboard.html",
        total_users=total_users,
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        cat_counts=cat_counts,
        avg_sleep=avg_sleep,
        avg_steps=avg_steps,
        avg_stress=avg_stress,
        avg_water=avg_water,
        avg_screen=avg_screen,
        comparison_table=comparison_table,
        best_model_name=best_model_name,
        trained_at=trained_at,
        feature_importances=feature_importances,
        min_required=MIN_RECORDS_FOR_ML,
    )


@app.route("/history")
def history():
    """Historical user records table with filtering and search."""
    df = load_all_users()
    users_list = []
    if not df.empty:
        # Sort by Timestamp descending or index
        users_list = df.to_dict(orient="records")
        users_list.reverse()
    return render_template("history.html", users=users_list)


@app.route("/report/<user_id>")
def download_report(user_id):
    """Generates and downloads the PDF wellness report."""
    user_data = get_user_by_id(user_id)
    if not user_data:
        flash(f"User with ID '{user_id}' not found.", "warning")
        return redirect(url_for("history"))

    pdf_path = generate_pdf_report(user_data)
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"WellCare_Report_{user_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/train", methods=["POST"])
def retrain():
    """Triggers ML model training and evaluation."""
    result = train_and_evaluate_models()
    if result["status"] == "success":
        flash(f"Success: {result['message']}", "success")
    elif result["status"] == "insufficient_data":
        flash(f"Notice: {result['message']}", "warning")
    else:
        flash(f"Training Notice: {result['message']}", "info")
    return redirect(url_for("dashboard"))


@app.route("/seed-demo", methods=["POST"])
def seed_demo():
    """Populates initial synthetic user records for academic demonstration."""
    count = generate_seed_dataset(n=60)
    train_and_evaluate_models()
    flash(f"Successfully generated {count} realistic demonstration user records and trained ML models!", "success")
    return redirect(url_for("dashboard"))


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Returns all registered users as JSON."""
    df = load_all_users()
    return jsonify({
        "status": "success",
        "count": len(df),
        "users": df.to_dict(orient="records") if not df.empty else [],
    })


@app.route("/api/user/<user_id>", methods=["GET"])
def api_get_user(user_id):
    """Returns a specific user's record, score, prediction, and recommendations."""
    user_data = get_user_by_id(user_id)
    if not user_data:
        return jsonify({"status": "error", "message": f"User ID '{user_id}' not found"}), 404

    prediction = predict_wellness(user_data)
    scoring = prediction["Scoring_Details"]
    recommendations = generate_recommendations(user_data, scoring)

    return jsonify({
        "status": "success",
        "user_id": user_id,
        "user_data": user_data,
        "prediction": prediction,
        "scoring": scoring,
        "recommendations": recommendations,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for scoring, ML prediction, and persisting a new record."""
    raw_data = request.get_json(force=True, silent=True) or request.form.to_dict()
    if not raw_data:
        return jsonify({"status": "error", "message": "No input payload provided."}), 400

    is_valid, err_msg, clean_data = validate_user_input(raw_data)
    if not is_valid:
        return jsonify({"status": "error", "message": err_msg}), 422

    save_user_record(clean_data)
    prediction = predict_wellness(clean_data)
    scoring = prediction["Scoring_Details"]
    recommendations = generate_recommendations(clean_data, scoring)

    return jsonify({
        "status": "success",
        "user_id": clean_data["User_ID"],
        "wellness_score": scoring["Wellness_Score"],
        "category": prediction["Predicted_Category"],
        "confidence": prediction["Confidence"],
        "model_used": prediction["Model_Name"],
        "recommendations": [act["title"] + ": " + act["action"] for act in recommendations["priority_actions"]] + [adv["title"] + ": " + adv["action"] for adv in recommendations["general_advice"]],
        "full_details": {
            "prediction": prediction,
            "scoring": scoring,
            "recommendations": recommendations,
        }
    })


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """Returns analytics and ML performance summary in JSON format."""
    df = load_all_users()
    model_bundle = load_model_bundle()
    
    return jsonify({
        "status": "success",
        "total_users": len(df),
        "average_wellness_score": round(float(df["Wellness_Score"].mean()), 1) if not df.empty and "Wellness_Score" in df.columns else 0.0,
        "category_distribution": df["Wellness_Category"].value_counts().to_dict() if not df.empty and "Wellness_Category" in df.columns else {},
        "model_performance": model_bundle.get("comparison", []) if model_bundle else [],
        "best_model": model_bundle.get("best_model_name", None) if model_bundle else None,
    })


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Triggers ML model training via API."""
    res = train_and_evaluate_models()
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
