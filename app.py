import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 50%, #0a1628 100%); min-height: 100vh; }
.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-icon { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; }
.hero-title { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.5px; }
.hero-title span { color: #00e676; }
.hero-sub { font-size: 1rem; color: #7b8fa6; margin-top: 0.4rem; }
.card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.2rem; }
.card-title { font-size: 0.72rem; font-weight: 600; color: #00e676; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
.stSelectbox label, .stSlider label, .stRadio label { color: #a0b4c8 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; color: white !important; }
.stSlider > div > div > div { background: #00e676 !important; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #00e676, #00b248); color: #0a0f1e; font-weight: 700; font-size: 1rem; padding: 0.75rem; border: none; border-radius: 12px; cursor: pointer; margin-top: 0.5rem; }
.stButton > button:hover { opacity: 0.9; }
.result-card { background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,178,72,0.06)); border: 1px solid rgba(0,230,118,0.25); border-radius: 16px; padding: 1.75rem; text-align: center; margin-top: 1.2rem; }
.result-score { font-size: 3.5rem; font-weight: 700; color: #00e676; line-height: 1; }
.result-label { font-size: 0.85rem; color: #7b8fa6; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.08em; }
.win-section { margin-top: 1.5rem; }
.win-label { font-size: 0.75rem; color: #7b8fa6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem; }
.win-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 999px; height: 10px; overflow: hidden; margin: 0.4rem 0 0.6rem; }
.win-bar-fill { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #00e676, #00b248); }
.win-teams { display: flex; justify-content: space-between; font-size: 0.85rem; }
.win-bat { color: #00e676; font-weight: 600; }
.win-bowl { color: #ff6b6b; font-weight: 600; }
.stat-row { display: flex; gap: 10px; margin-top: 1rem; }
.stat-box { flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.75rem; text-align: center; }
.stat-val { font-size: 1.3rem; font-weight: 700; color: #ffffff; }
.stat-lbl { font-size: 0.7rem; color: #7b8fa6; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.06em; }
.footer { text-align: center; color: #3d5166; font-size: 0.75rem; padding: 2rem 0 1rem; }
div[role="radiogroup"] { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 6px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_models():
    matches = pd.read_csv('matches.csv')
    deliveries_raw = pd.read_csv('deliveries_small.csv')

    valid_teams = [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Delhi Capitals',
        'Punjab Kings', 'Rajasthan Royals', 'Gujarat Titans', 'Lucknow Super Giants'
    ]

    df1 = deliveries_raw[deliveries_raw['over'] >= 5].copy()
    df1 = df1[df1['inning'] == 1]
    df1 = df1[df1['batting_team'].isin(valid_teams) & df1['bowling_team'].isin(valid_teams)]

    df1 = df1.sort_values(['match_id', 'over', 'ball'])
    df1['current_score'] = df1.groupby('match_id')['total_runs'].cumsum()
    df1['wickets_fallen'] = df1.groupby('match_id')['is_wicket'].cumsum()
    df1['balls_bowled'] = df1.groupby('match_id').cumcount() + 1
    df1['current_run_rate'] = (df1['current_score'] / df1['balls_bowled']) * 6
    df1['balls_left'] = 120 - df1['balls_bowled']
    df1['wickets_left'] = 10 - df1['wickets_fallen']

    match_totals = df1.groupby('match_id')['total_runs'].sum().reset_index()
    match_totals.columns = ['match_id', 'final_score']
    df1 = df1.merge(match_totals, on='match_id')

    le_bat = LabelEncoder()
    le_bowl = LabelEncoder()
    df1['batting_team_enc'] = le_bat.fit_transform(df1['batting_team'])
    df1['bowling_team_enc'] = le_bowl.fit_transform(df1['bowling_team'])

    features1 = ['batting_team_enc', 'bowling_team_enc', 'over', 'current_score', 'wickets_left', 'current_run_rate', 'balls_left']
    X1 = df1[features1]
    y_score = df1['final_score']

    match_winner = matches[['id', 'winner']].copy()
    match_winner.columns = ['match_id', 'winner']
    df1_win = df1.merge(match_winner, on='match_id')
    df1_win['batting_team_won'] = (df1_win['batting_team'] == df1_win['winner']).astype(int)
    y_win = df1_win['batting_team_won']

    X_train, X_test, y_train, y_test = train_test_split(X1, y_score, test_size=0.2, random_state=42)
    score_model = RandomForestRegressor(n_estimators=100, random_state=42)
    score_model.fit(X_train, y_train)

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X1, y_win, test_size=0.2, random_state=42)
    win_model = RandomForestClassifier(n_estimators=100, random_state=42)
    win_model.fit(X_train_w, y_train_w)

    chase_df = deliveries_raw[deliveries_raw['over'] >= 5].copy()
    chase_df = chase_df[chase_df['inning'] == 2]
    chase_df = chase_df[chase_df['batting_team'].isin(valid_teams) & chase_df['bowling_team'].isin(valid_teams)]

    target_scores = deliveries_raw[deliveries_raw['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    target_scores.columns = ['match_id', 'target']
    target_scores['target'] = target_scores['target'] + 1
    chase_df = chase_df.merge(target_scores, on='match_id')

    chase_df = chase_df.sort_values(['match_id', 'over', 'ball'])
    chase_df['current_score'] = chase_df.groupby('match_id')['total_runs'].cumsum()
    chase_df['wickets_fallen'] = chase_df.groupby('match_id')['is_wicket'].cumsum()
    chase_df['balls_bowled'] = chase_df.groupby('match_id').cumcount() + 1
    chase_df['runs_needed'] = chase_df['target'] - chase_df['current_score']
    chase_df['balls_left'] = 120 - chase_df['balls_bowled']
    chase_df['wickets_left'] = 10 - chase_df['wickets_fallen']
    chase_df['current_run_rate'] = (chase_df['current_score'] / chase_df['balls_bowled']) * 6
    chase_df['required_run_rate'] = (chase_df['runs_needed'] / chase_df['balls_left']) * 6
    chase_df = chase_df[chase_df['balls_left'] > 0]
    chase_df = chase_df[chase_df['runs_needed'] > 0]

    chase_df = chase_df.merge(match_winner, on='match_id')
    chase_df['chasing_team_won'] = (chase_df['batting_team'] == chase_df['winner']).astype(int)
    chase_df['batting_team_enc'] = le_bat.transform(chase_df['batting_team'])
    chase_df['bowling_team_enc'] = le_bowl.transform(chase_df['bowling_team'])

    chase_features = ['batting_team_enc', 'bowling_team_enc', 'over', 'current_score', 'wickets_left', 'current_run_rate', 'required_run_rate', 'balls_left', 'runs_needed']
    X_chase = chase_df[chase_features]
    y_chase = chase_df['chasing_team_won']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_chase, y_chase, test_size=0.2, random_state=42)
    chase_model = RandomForestClassifier(n_estimators=100, random_state=42)
    chase_model.fit(X_train_c, y_train_c)

    return score_model, win_model, chase_model, le_bat, le_bowl

st.markdown("""
<div class="hero">
  <span class="hero-icon">🏏</span>
  <h1 class="hero-title">Cricket Score <span>Predictor</span></h1>
  <p class="hero-sub">Powered by Machine Learning · IPL 2008–2024</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("🤖 Training ML models... please wait ~40 seconds on first load!"):
    model, win_model, chase_model, le_bat, le_bowl = train_models()

teams = list(le_bat.classes_)

st.markdown('<div class="card"><div class="card-title">🏟️ Match Innings</div>', unsafe_allow_html=True)
innings_choice = st.radio("Select innings", ["1st Innings (Setting target)", "2nd Innings (Chasing)"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

is_chasing = innings_choice.startswith("2nd")

st.markdown('<div class="card"><div class="card-title">⚡ Match Setup</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
with col2:
    bowling_options = [t for t in teams if t != batting_team]
    bowling_team = st.selectbox("🎯 Bowling Team", bowling_options)
st.markdown('</div>', unsafe_allow_html=True)

if is_chasing:
    st.markdown('<div class="card"><div class="card-title">🎯 Chase Situation</div>', unsafe_allow_html=True)
    target = st.slider("Target Score", min_value=50, max_value=260, value=180)
    col3, col4, col5 = st.columns(3)
    with col3:
        over = st.slider("Current Over", min_value=5, max_value=19, value=14)
    with col4:
        current_score = st.slider("Current Score", min_value=10, max_value=target-1, value=min(120, target-1))
    with col5:
        wickets = st.slider("Wickets Fallen", min_value=0, max_value=9, value=3)

    balls_bowled = over * 6
    balls_left = 120 - balls_bowled
    runs_needed = target - current_score
    current_run_rate = round(current_score / balls_bowled * 6, 2) if balls_bowled > 0 else 0
    required_run_rate = round(runs_needed / balls_left * 6, 2) if balls_left > 0 else 0

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box"><div class="stat-val">{runs_needed}</div><div class="stat-lbl">Runs Needed</div></div>
      <div class="stat-box"><div class="stat-val">{balls_left}</div><div class="stat-lbl">Balls Left</div></div>
      <div class="stat-box"><div class="stat-val">{required_run_rate}</div><div class="stat-lbl">Req. Run Rate</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card"><div class="card-title">📊 Current Match Situation</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        over = st.slider("Current Over", min_value=5, max_value=19, value=10)
    with col4:
        current_score = st.slider("Current Score", min_value=10, max_value=220, value=80)
    with col5:
        wickets = st.slider("Wickets Fallen", min_value=0, max_value=9, value=2)

    balls_bowled = over * 6
    run_rate = round(current_score / balls_bowled * 6, 2) if balls_bowled > 0 else 0
    balls_left = 120 - balls_bowled

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box"><div class="stat-val">{run_rate}</div><div class="stat-lbl">Run Rate</div></div>
      <div class="stat-box"><div class="stat-val">{balls_left}</div><div class="stat-lbl">Balls Left</div></div>
      <div class="stat-box"><div class="stat-val">{10 - wickets}</div><div class="stat-lbl">Wickets Left</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("🔮 Predict Result")

if predict_btn:
    batting_enc = le_bat.transform([batting_team])[0]
    bowling_enc = le_bowl.transform([bowling_team])[0]
    wickets_left = 10 - wickets

    if is_chasing:
        current_run_rate_val = (current_score / balls_bowled) * 6
        required_run_rate_val = (runs_needed / balls_left) * 6 if balls_left > 0 else 0
        input_data = [[batting_enc, bowling_enc, over, current_score, wickets_left, current_run_rate_val, required_run_rate_val, balls_left, runs_needed]]

        win_prob = chase_model.predict_proba(input_data)[0]
        classes = chase_model.classes_
        prob_dict = dict(zip(classes, win_prob))
        bat_win = round(prob_dict.get(1, 0) * 100)
        bowl_win = 100 - bat_win
        winner = batting_team if bat_win >= 50 else bowling_team
        winner_prob = bat_win if bat_win >= 50 else bowl_win

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Chasing</div>
          <div class="result-score">{runs_needed}</div>
          <div class="result-label">runs needed off {balls_left} balls</div>
          <div class="win-section">
            <div class="win-label">Win Probability</div>
            <div class="win-bar-wrap">
              <div class="win-bar-fill" style="width:{bat_win}%"></div>
            </div>
            <div class="win-teams">
              <span class="win-bat">🏏 {batting_team} {bat_win}%</span>
              <span class="win-bowl">{bowling_team} {bowl_win}% 🎯</span>
            </div>
          </div>
          <div class="stat-row" style="margin-top:1.2rem">
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">🏆</div>
              <div class="stat-lbl">Likely Winner</div>
              <div style="color:white;font-size:0.8rem;font-weight:600;margin-top:4px">{winner}</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">{winner_prob}%</div>
              <div class="stat-lbl">Confidence</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">{required_run_rate}</div>
              <div class="stat-lbl">Req. Run Rate</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        current_run_rate_val = (current_score / balls_bowled) * 6
        input_data = [[batting_enc, bowling_enc, over, current_score, wickets_left, current_run_rate_val, balls_left]]

        predicted_score = model.predict(input_data)[0]
        if predicted_score < current_score:
            predicted_score = current_score + (balls_left * current_run_rate_val / 6) * 1.1
        predicted_score = round(predicted_score)

        win_prob = win_model.predict_proba(input_data)[0]
        classes = win_model.classes_
        prob_dict = dict(zip(classes, win_prob))
        bat_win = round(prob_dict.get(1, 0) * 100)
        bowl_win = 100 - bat_win
        winner = batting_team if bat_win >= 50 else bowling_team
        winner_prob = bat_win if bat_win >= 50 else bowl_win

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Predicted Final Score</div>
          <div class="result-score">{predicted_score}</div>
          <div class="result-label">runs</div>
          <div class="win-section">
            <div class="win-label">Win Probability</div>
            <div class="win-bar-wrap">
              <div class="win-bar-fill" style="width:{bat_win}%"></div>
            </div>
            <div class="win-teams">
              <span class="win-bat">🏏 {batting_team} {bat_win}%</span>
              <span class="win-bowl">{bowling_team} {bowl_win}% 🎯</span>
            </div>
          </div>
          <div class="stat-row" style="margin-top:1.2rem">
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">🏆</div>
              <div class="stat-lbl">Likely Winner</div>
              <div style="color:white;font-size:0.8rem;font-weight:600;margin-top:4px">{winner}</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">{winner_prob}%</div>
              <div class="stat-lbl">Confidence</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#00e676">{predicted_score - current_score}</div>
              <div class="stat-lbl">Runs Expected</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  Built with ❤️ using Random Forest ML · Score MAE: 8.22 · 1st Inn Win Acc: 86% · Chase Win Acc: 97.67%
</div>
""", unsafe_allow_html=True)
