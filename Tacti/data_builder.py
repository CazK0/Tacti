import pandas as pd

url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values(by='Date')


def get_form(team, date, data):
    past_games = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)].tail(3)

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


h_pts, h_gs, h_gc = [], [], []
a_pts, a_gs, a_gc = [], [], []

for index, row in df.iterrows():
    hp, hg, hgc = get_form(row['HomeTeam'], row['Date'], df)
    ap, ag, agc = get_form(row['AwayTeam'], row['Date'], df)

    h_pts.append(hp)
    h_gs.append(hg)
    h_gc.append(hgc)

    a_pts.append(ap)
    a_gs.append(ag)
    a_gc.append(agc)

df['H_FormPts'] = h_pts
df['H_FormGS'] = h_gs
df['H_FormGC'] = h_gc

df['A_FormPts'] = a_pts
df['A_FormGS'] = a_gs
df['A_FormGC'] = a_gc

df.to_csv('match_features.csv', index=False)
print("Head-to-Head Data pipeline complete. Output saved to match_features.csv")
