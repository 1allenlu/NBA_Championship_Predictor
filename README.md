# Predictive Analytics System for NBA Roster Success

This project is an end-to-end sports analytics system that predicts the probability of an NBA team winning the championship based on a user-selected roster.  
It combines **machine learning**, **deep learning**, and an interactive **Streamlit** interface to allow users to build teams under salary-cap constraints and receive real-time predictions.

---

## 🏀 What the Project Does
- **Data Preprocessing**: Cleans, aggregates, and formats NBA player statistics with salary data.
- **Feature Engineering**: Aggregates individual player features into team-level metrics.
- **Model Training**: Trains multiple models (Logistic Regression, Random Forest, XGBoost, PyTorch Neural Network) with hyperparameter tuning.
- **Probability Calibration**: Improves reliability of predicted probabilities using Platt scaling/Isotonic regression.
- **Soft-Vote Ensembling**: Combines the best models to improve accuracy and F1 score.
- **Interactive App**: Streamlit-based UI where users build rosters and see predictions instantly.
- **AWS S3 Integration**: Stores and retrieves trained models from the cloud for flexible deployment.

---

## 🔄 Pipeline Overview
1. **Data Acquisition & Cleaning** (`src/data/`):
   - Collect player stats and salaries.
   - Handle missing values and format columns.
   
2. **Feature Engineering** (`src/user_team_simulator/roster_features.py`):
   - Aggregate player-level features into a single team vector.
   - Create advanced metrics for better predictive power.

3. **Model Training & Evaluation** (`src/user_team_simulator/train_roster.py`):
   - Train multiple ML/DL models.
   - Evaluate on accuracy, F1 score, and calibration metrics.

4. **Probability Calibration & Ensembling** (`src/user_team_simulator/calibrate_and_ensemble.py`):
   - Apply calibration techniques to improve probability estimates.
   - Combine models via soft-vote ensembling.

5. **Deployment via Streamlit** (`frontend/team_builder.py`):
   - Users select players within a salary cap.
   - Backend aggregates features, loads the trained model, and outputs championship probability.
   - Supports both **local models** and **models from AWS S3**.

---

## 📂 Folder Structure
```
├── data/                         # Processed datasets
│   ├── raw/                      # Raw CSV files
│   ├── processed/                # Cleaned and feature-engineered CSV files
│   └── testing/                  # Test files
├── models/                       # Trained models (local storage)
├── src/
│   ├── historial_preductor/      # Data cleaning scripts
│   ├── user_team_simulator/      # Feature engineering, training, calibration, and 
├── frontend/
│   ├── team_builder.py           # Streamlit app for user interaction
│   └── historial_preductor.py    # Streamlit app for user interaction
├── results/                      # Model evaluation 
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```
---

**Tech Stack:** Python · PyTorch · scikit-learn · XGBoost · Pandas · Streamlit · AWS S3  
**Key Skills:** Machine Learning, Deep Learning, Data Preprocessing, Feature Engineering, Model Calibration, Ensembling, Cloud Model Hosting