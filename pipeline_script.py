import pandas as pd
import numpy as np
import joblib
import logging
import sqlite3
import os
import sys
from datetime import datetime
import google.generativeai as genai
import json
from dotenv import load_dotenv
load_dotenv()

# =========================
# Logging Configuration
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)   # ← important
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tgt_regression_pipeline.joblib")
DATA_PATH = os.path.join(BASE_DIR, "CleanedData", "cleaned_aircraft_sensor_data.csv")
DB_PATH = os.path.join(BASE_DIR, "Database", "engine_data.db")


FEATURE_LIST = [
    "EPR","AFT","OIP","OIT","P160","P50","P3",
    "T2","T25","T3","TCAF","TCAR","TN",
    "N1","N2","N3","MN","FF"
]


# =========================
# Step 1: Load Model
# =========================
def load_model():
    logger.info("Loading trained regression pipeline...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    return model


# =========================
# Step 2: Load Data (via SQLite for realism)
# =========================
def load_data():
    logger.info("Loading cleaned dataset...")

    df = pd.read_csv(DATA_PATH)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("engine_data", conn, if_exists="replace", index=False)

    df_db = pd.read_sql("SELECT * FROM engine_data", conn)
    conn.close()

    logger.info("Data loaded from SQLite successfully.")
    return df_db

def validate_data(df):
    logger.info("Validating input dataset...")

    required_columns = FEATURE_LIST + ["engine no", "datetime", "TGT"]

    # Check missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError("Dataset missing required columns.")

    # Check null percentage
    null_percent = df[FEATURE_LIST].isnull().mean() * 100
    high_null = null_percent[null_percent > 5]

    if not high_null.empty:
        logger.warning(f"Features with >5% missing values:\n{high_null}")


    logger.info("Data validation completed successfully.")


# =========================
# Step 3: Predict TGT
# =========================
def predict_tgt(model, df):
    logger.info("Running TGT predictions...")
    X = df[FEATURE_LIST]
    df["predicted_TGT"] = model.predict(X)
    logger.info("Prediction completed.")
    return df


# =========================
# Step 4: Compute Residuals
# =========================
def compute_residuals(df):
    logger.info("Computing residuals...")
    df["residual"] = df["TGT"] - df["predicted_TGT"]
    logger.info("Residual computation completed.")
    return df


# =========================
# Step 5: Engine-Level Aggregation
# =========================
def compute_engine_stats(df):

    logger.info("Computing engine-level statistics...")

    engine_stats = (
        df.groupby("engine no")["residual"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    # Fleet statistics
    fleet_mean = engine_stats["mean"].mean()
    fleet_std = engine_stats["mean"].std()

    engine_stats["z_score"] = (
        (engine_stats["mean"] - fleet_mean) / fleet_std
    )

    # Persistence
    threshold = 2 * df["residual"].std()
    df["high_residual"] = df["residual"] > threshold

    persistence = (
        df.groupby("engine no")["high_residual"]
        .mean()
        .reset_index()
        .rename(columns={"high_residual": "persistence"})
    )

    engine_stats = engine_stats.merge(persistence, on="engine no")

    logger.info("Engine-level anomaly statistics computed.")

    return engine_stats


# =========================
# Step 6: Identify Affected Engines
# =========================
def identify_anomalies(engine_stats):

    logger.info("Identifying affected engines...")

    flagged = engine_stats[
        (engine_stats["z_score"] > 2) &
        (engine_stats["persistence"] > 0.5)
    ].sort_values("z_score", ascending=False)

    logger.info(f"Flagged engines: {flagged['engine no'].tolist()}")

    return flagged


# =========================
# Step 7: LLM Summary
# =========================


def generate_llm_summary(flagged_engines):

    logger.info("Generating structured LLM engineering summary using Gemini...")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        raise ValueError("Missing Gemini API Key.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an aircraft engine performance diagnostics assistant.

Based on the following engine anomaly statistics:

{flagged_engines.to_string(index=False)}

Return ONLY valid JSON in the following format:

{{
  "critical_engines": [list of engine numbers ranked by severity],
  "risk_assessment": "Low / Moderate / High",
  "engineering_summary": "Concise explanation of anomaly behavior",
  "recommended_action": "Recommended inspection or follow-up action"
}}

Do NOT include any text outside JSON.

In enginnering summary only add which are all the anomaly engines and why and what steps can be done.
"""

    response = model.generate_content(prompt)

    try:
        structured_output = json.loads(response.text)
        logger.info("Structured LLM summary generated successfully.")
        return structured_output

    except json.JSONDecodeError:
        logger.warning("LLM response was not valid JSON. Returning raw text.")
        return {
            "critical_engines": [],
            "risk_assessment": "Unknown",
            "engineering_summary": response.text,
            "recommended_action": "Manual review required."
        }
# =========================
# Main Execution
# =========================
def main():

    logger.info("===== TGT Anomaly Detection Pipeline Started =====")

    model = load_model()
    df = load_data()
    
    validate_data(df)
    
    df = predict_tgt(model, df)
    df = compute_residuals(df)

    engine_stats = compute_engine_stats(df)
    flagged = identify_anomalies(engine_stats)

    print("\n===== FLAGGED ENGINES =====")
    print(flagged)

    if not flagged.empty:
        summary = generate_llm_summary(flagged)

        print("\n===== LLM STRUCTURED ENGINEERING SUMMARY =====\n")
        print(f"Critical Engines: {summary['critical_engines']}")
        print(f"Risk Level: {summary['risk_assessment']}")
        print(f"\nEngineering Summary:\n{summary['engineering_summary']}")
        print(f"\nRecommended Action:\n{summary['recommended_action']}")

    logger.info("===== Pipeline Execution Completed =====")


if __name__ == "__main__":
    main()
