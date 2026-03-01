import pandas as pd

seasons = ["2122", "2223", "2324", "2425", "2526"]
dfs = []

for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    try:
        temp_df = pd.read_csv(url)
        temp_df = temp_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], dayfirst=True)
        dfs.append(temp_df)
    except Exception as e:
        pass

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by='Date')
df = df.dropna(subset=['FTR'])

def get_ewma_form(team, date, data):
    past_games = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)].tail(5)
    
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

h_pts, h_gs, h_gc = [], [], []
a_pts, a_gs, a_gc = [], [], []

for index, row in df.iterrows():
    hp, hg, hgc = get_ewma_form(row['HomeTeam'], row['Date'], df)
    ap, ag, agc = get_ewma_form(row['AwayTeam'], row['Date'], df)
    
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

df = df[(df['H_FormPts'] > 0) & (df['A_FormPts'] > 0)]

df.to_csv('match_features.csv', index=False)
