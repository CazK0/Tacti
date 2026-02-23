import pandas as pd
import numpy as np

url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

columns_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
df = df[columns_to_keep]

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

df['HomePoints'] = np.where(df['FTR'] == 'H', 3, np.where(df['FTR'] == 'D', 1, 0))
df['AwayPoints'] = np.where(df['FTR'] == 'A', 3, np.where(df['FTR'] == 'D', 1, 0))

home_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HomePoints']].copy()
home_df.columns = ['Date', 'Team', 'Opponent', 'GoalsScored', 'GoalsConceded', 'Points']
home_df['Venue'] = 'Home'

away_df = df[['Date', 'AwayTeam', 'HomeTeam', 'FTAG', 'FTHG', 'AwayPoints']].copy()
away_df.columns = ['Date', 'Team', 'Opponent', 'GoalsScored', 'GoalsConceded', 'Points']
away_df['Venue'] = 'Away'

team_history = pd.concat([home_df, away_df]).sort_values(by=['Team', 'Date'])

team_history['FormPoints'] = team_history.groupby('Team')['Points'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
team_history['FormGoalsScored'] = team_history.groupby('Team')['GoalsScored'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_history['FormGoalsConceded'] = team_history.groupby('Team')['GoalsConceded'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

team_history = team_history.dropna()

team_history.to_csv('team_features.csv', index=False)
print("Data pipeline complete. Output saved to team_features.csv")