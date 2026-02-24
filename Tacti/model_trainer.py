import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('match_features.csv')

df['Target_Win'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)

features = ['H_FormPts', 'H_FormGS', 'H_FormGC', 'A_FormPts', 'A_FormGS', 'A_FormGC']
X = df[features]
y = df['Target_Win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model trained. Accuracy on unseen test data: {accuracy * 100:.2f}%")

joblib.dump(model, 'cortex_model.pkl')
