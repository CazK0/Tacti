from flask import Flask, render_template, request
import pandas as pd
import joblib

print("Loading AI model...")
model = joblib.load('cortex_model.pkl')

print("Loading data...")
df = pd.read_csv('match_features.csv')

print("Extracting current teams...")
teams = sorted(df['HomeTeam'].drop_duplicates(keep='last').tail(20).tolist())

app = Flask(__name__)

def get_latest_form(team, data):
    past_games = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)].tail(5)
    
    if len(past_games) < 5:
        return 0, 0, 0
    
    pts = 0
    gs = 0
    gc = 0
    weights = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    for i, (_, row) in enumerate(past_games.iterrows()):
        w = weights[i]
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H': pts += 3 * w
            elif row['FTR'] == 'D': pts += 1 * w
            gs += row['FTHG'] * w
            gc += row['FTAG'] * w
        else:
            if row['FTR'] == 'A': pts += 3 * w
            elif row['FTR'] == 'D': pts += 1 * w
            gs += row['FTAG'] * w
            gc += row['FTHG'] * w
            
    return pts, gs, gc

@app.route('/', methods=['GET', 'POST'])
def index():
    home_prob = None
    draw_prob = None
    away_prob = None
    home_team = None
    away_team = None

    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')

        if home_team and away_team and home_team != away_team:
            h_pts, h_gs, h_gc = get_latest_form(home_team, df)
            a_pts, a_gs, a_gc = get_latest_form(away_team, df)
            
            features = [[h_pts, h_gs, h_gc, a_pts, a_gs, a_gc]]
            
            probs = model.predict_proba(features)[0]
            
            away_prob = round(probs[0] * 100, 1)
            draw_prob = round(probs[1] * 100, 1)
            home_prob = round(probs[2] * 100, 1)

    return render_template('index.html', teams=teams, home_prob=home_prob, draw_prob=draw_prob, away_prob=away_prob, home_team=home_team, away_team=away_team)

if __name__ == '__main__':
    print("Starting Flask server on port 5005...")
    app.run(debug=True, port=5005)
