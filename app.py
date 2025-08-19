from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get values from form
        study_hours = int(request.form["study_hours"])
        sleep_hours = int(request.form["sleep_hours"])
        participation = int(request.form["participation"])
        activities = int(request.form["activities"])
        internet_usage = int(request.form["internet_usage"])
        motivation = int(request.form["motivation"])

        # arrange into numpy array
        features = np.array([[study_hours, sleep_hours, participation,
                              activities, internet_usage, motivation]])

        # model prediction (numeric score)
        score = model.predict(features)[0]

        # Convert score → performance category with emojis
        if score >= 85:
            performance = "Excellent 🌟"
        elif 70 <= score < 85:
            performance = "Good 🙂"
        elif 50 <= score < 70:
            performance = "Average 😐"
        else:
            performance = "Needs Improvement ⚠️"

        return render_template("result.html",
                               prediction=performance,
                               score=round(score, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
