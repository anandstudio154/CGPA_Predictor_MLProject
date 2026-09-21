from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("cgpa_model.pkl")


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    sleep_hours = float(request.form["sleep_hours"])
    social_hours = float(request.form["social_hours"])

    features = [[
        age,
        attendance,
        study_hours,
        sleep_hours,
        social_hours
    ]]

    prediction = model.predict(features)[0]
    prediction = prediction * 2.5

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)