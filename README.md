# 🏏 Cricket Score & Win Predictor

An end-to-end Machine Learning web app that predicts IPL match outcomes in real time — final score, 1st innings win probability, and 2nd innings chase win probability — built with Python, scikit-learn, and Streamlit.

**🔗 Live App:** [cricket-score-predictor-ld5wend6tsxf2wcnc8appvv.streamlit.app](https://cricket-score-predictor-ld5wend6tsxf2wcnc8appvv.streamlit.app/)

---

## 📌 Overview

This project predicts what will happen in a live IPL match based on the current match situation — over, score, wickets, and teams. It uses three separate Random Forest models trained on ball-by-ball IPL data (2008–2024):

| Model | Purpose | Performance |
|---|---|---|
| Score Predictor | Predicts final 1st innings score | MAE: 8.22 runs |
| 1st Innings Win Model | Predicts win probability while batting first | 86% accuracy |
| Chase Win Model | Predicts win probability while chasing a target | 97.67% accuracy |

---

## ✨ Features

- 🏏 Select any two IPL teams (batting and bowling)
- 📊 Adjust current over, score, and wickets fallen using sliders
- 🎯 Toggle between 1st innings and 2nd innings (chase) mode
- 🔮 Get instant predictions: final score, win probability, and likely winner
- 📈 Chase mode shows runs needed, required run rate, and live win probability
- 🎨 Clean, dark cricket-themed responsive UI

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Library:** scikit-learn (Random Forest Regressor & Classifier)
- **Data Processing:** pandas, numpy
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Data Source:** [IPL Complete Dataset (Kaggle)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 📂 Project Structure

```
cricket-score-predictor/
├── app.py                  # Streamlit web app (trains models live from CSVs)
├── requirements.txt        # Python dependencies
├── matches.csv             # Match-level IPL data (2008-2024)
├── deliveries_small.csv    # Ball-by-ball delivery data (compressed)
└── README.md
```

---

## ⚙️ How It Works

### 1. Data cleaning
Raw ball-by-ball data is filtered to remove the first 5 overs (too early to predict), keep only current IPL teams, and drop irrelevant columns.

### 2. Feature engineering
Key features are engineered from raw deliveries:
- Current run rate
- Wickets left
- Balls left
- For chases: target score, runs needed, required run rate

### 3. Model training
Three Random Forest models are trained:
- **Regressor** for final score prediction (1st innings)
- **Classifier** for win probability (1st innings)
- **Classifier** for chase win probability (2nd innings), using target and required run rate as additional signals

### 4. Deployment
The Streamlit app loads the raw CSVs and retrains all three models on startup (cached with `@st.cache_resource` so it only happens once per session), then serves live predictions through an interactive UI.

---

## 🚀 Running Locally

```bash
git clone https://github.com/sakthi0015/cricket-score-predictor.git
cd cricket-score-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📊 Model Performance Details

**Score Prediction (1st Innings)**
- Algorithm: Random Forest Regressor (100 trees)
- Mean Absolute Error: 8.22 runs
- R² Score: 0.75

**Win Probability (1st Innings)**
- Algorithm: Random Forest Classifier (100 trees)
- Accuracy: 86%

**Chase Win Probability (2nd Innings)**
- Algorithm: Random Forest Classifier (100 trees)
- Accuracy: 97.67%
- Higher accuracy is expected here since required run rate becomes a near-deterministic signal as the chase progresses

---

## 🔮 Future Improvements

- Add player-level features (batting/bowling strike rates, form)
- Include venue-specific pitch behavior
- Add live data integration for ongoing matches
- Experiment with gradient boosting models (XGBoost, LightGBM) for improved accuracy

---

## 🙋 About

Built by Sakthi as a hands-on Machine Learning learning project — covering the full ML lifecycle from raw data to a deployed, publicly usable product.

⭐ If you found this project interesting, consider starring the repo!
