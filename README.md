# ✈️ Aircraft Engine TGT Anomaly Detection System

🔗 Live Application:  
https://aircraft-tgt-anomaly-detection.streamlit.app/

---

## 📌 Problem Statement

During engine development, a fault may cause certain aircraft engines to operate at **higher-than-expected Turbine Gas Temperature (TGT)** during cruise phase.

We are provided with time-series cruise-phase sensor snapshots across multiple engines.

**Objective:**  
Identify engines that are systematically operating hotter than expected under comparable operating conditions.

---

## 🧠 Approach Overview

Since no anomaly labels are available, a supervised classification approach is not feasible.

Instead, this solution implements a:

### 🔍 Regression-Based Residual Anomaly Detection Framework  
(Industry term: *Model-Based Anomaly Detection*)

Core idea:
1. Learn expected TGT as a function of operating parameters.
2. Compute deviation (residual) from expected behavior.
3. Identify engines with statistically significant and sustained positive deviations.

---

## 🔬 Methodology

### 1️⃣ Data Understanding & Cleaning
- Removed columns with excessive missingness.
- Median imputation for minor missing values.
- Engine-wise train-test split to prevent leakage.
- StandardScaler applied for regression stability.

---

### 2️⃣ Modeling Expected Thermal Behavior

Models evaluated:
- Linear Regression
- Random Forest Regressor

Final selection:
- **Linear Regression** (better generalization across engines)

Residual defined as:
  Residual = Actual TGT − Predicted TGT


Positive residual → engine running hotter than expected.

---

### 3️⃣ Fleet-Level Statistical Normalization

Engine-level residual means were compared against fleet statistics.

Z-score computed as:
  Z = (Engine Mean Residual − Fleet Mean) / Fleet Std


Threshold:
- Z-score > 2 → statistically significant deviation.

---

### 4️⃣ Persistence Validation

To avoid flagging engines based on isolated spikes:

Persistence =
% of observations where residual > 2 × residual_std

This ensures anomalies are structural rather than transient.

---

## 🚨 Identified Affected Engines

| Engine | Z-Score | Persistence | Severity |
|--------|---------|------------|----------|
| 149    | ~2.9    | 84%        | High     |
| 131    | >2.0    | ~59%       | Moderate |
| 125    | >2.0    | ~58%       | Moderate |

Engine 149 exhibits a sustained and statistically significant overheating pattern.

---

## 🏭 Industry Context

This approach is commonly used in:

- Aircraft engine health monitoring
- Predictive maintenance systems
- Industrial turbine diagnostics
- Digital twin residual monitoring
- Statistical process control with model-based baselines

The framework aligns with modern condition-based maintenance strategies.

---

## 🖥 Interactive Dashboard

The Streamlit dashboard provides:

- Fleet-level anomaly overview
- Engine severity ranking
- Time-series residual trend visualization
- LLM-generated structured engineering summary
- Recommended inspection actions

🔗 https://aircraft-tgt-anomaly-detection.streamlit.app/

---

## ⚙️ Operationalization

This solution is formalized into a deployable pipeline:

- Model serialized using `joblib`
- Standalone inference script (`pipeline_script.py`)
- SQLite database integration
- Structured logging (Python `logging`)
- LLM integration (Gemini) for automated engineering assessment
- Interactive Streamlit dashboard

Designed for reproducibility and future production integration.

---

## 📁 Repository Structure
  Aircraft-TGT-Anomaly-Detection/
  │
  ├── data/
  ├── models/
  ├── notebook/
  ├── app.py
  ├── pipeline_script.py
  ├── requirements.txt
  └── README.md


---

## ⚠ Limitations

- Assumes majority of engines are healthy.
- Cruise-phase only (no takeoff/climb analysis).
- Does not incorporate physics-based thermal modeling.

---

## 🚀 Future Improvements

- Time-series drift detection (CUSUM / Bayesian change detection)
- Physics-informed modeling
- Integration with maintenance logs
- Real-time alerting system

---

## 👤 Author

R. Shrinivass  
MSc Data Science  
Regression-Based Anomaly Detection & Predictive Maintenance Systems
