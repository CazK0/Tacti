from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('cortex_model.pkl')
df = pd.read_csv('match_features.csv')

teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))


def get_latest_form(team, data):
    past_games = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)].tail(3)

    if len(past_games) == 0:
        return 0, 0, 0

    pts = 0
    gs = 0
    gc = 0

    for _, row in past_games.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H':
                pts += 3
            elif row['FTR'] == 'D':
                pts += 1
            gs += row['FTHG']
            gc += row['FTAG']
        else:
            if row['FTR'] == 'A':
                pts += 3
            elif row['FTR'] == 'D':
                pts += 1
            gs += row['FTAG']
            gc += row['FTHG']

    return pts, gs / len(past_games), gc / len(past_games)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    home_team = None
    away_team = None

    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')

        if home_team and away_team and home_team != away_team:
            h_pts, h_gs, h_gc = get_latest_form(home_team, df)
            a_pts, a_gs, a_gc = get_latest_form(away_team, df)

            features = [[h_pts, h_gs, h_gc, a_pts, a_gs, a_gc]]

            prob = model.predict_proba(features)[0][1]
            prediction = round(prob * 100, 1)

    return render_template('index.html', teams=teams, prediction=prediction, home_team=home_team, away_team=away_team)


if __name__ == '__main__':
    app.run(debug=True, port=5000)