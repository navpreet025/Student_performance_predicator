import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import pickle

# Example dataset (replace with real data later)
data = {
    "study_hours": [2, 5, 7, 3, 4, 6, 8],
    "sleep_hours": [7, 6, 5, 8, 7, 6, 5],
    "participation": [5, 8, 9, 4, 6, 7, 10],
    "extracurricular": [1, 2, 0, 3, 1, 2, 1],
    "internet_usage": [3, 5, 6, 2, 4, 5, 7],
    "motivation_level": [6, 7, 8, 5, 6, 7, 9],
    "score": [55, 75, 90, 50, 65, 80, 95]
}

df = pd.DataFrame(data)

# Features & target
X = df.drop("score", axis=1)
y = df["score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Assuming your trained model is called `model`
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


print("✅ Model trained and saved!")
