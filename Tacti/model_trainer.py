import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('match_features.csv')

target_mapping = {'A': 0, 'D': 1, 'H': 2}
df['FTR_Numeric'] = df['FTR'].map(target_mapping)

features = ['H_FormPts', 'H_FormGS', 'H_FormGC', 'A_FormPts', 'A_FormGS', 'A_FormGC']
X = df[features]
y = df['FTR_Numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"XGBoost Model trained. Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'cortex_model.pkl')
