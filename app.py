from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("models/Insurance_predictor_final.sav", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    bmi = float(request.form["bmi"])
    Child = int(request.form["Child"])
    Gender = int(request.form["Gender"])
    smoker = float(request.form["smoker"])

    #data = np.array([[bgr, bu, sc, pcv, wc]])

    data = pd.DataFrame(
    np.array([[age, bmi, Child, Gender, smoker]]),
    columns=["age", "bmi", "Child", "Gender", "smoker"]
)
    
    result=model.predict([[age,bmi,Child,Gender,smoker]])
    result = round(float(result[0]), 2)

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)