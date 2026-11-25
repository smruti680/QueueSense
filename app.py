"""app.py - QueueSense Flask web application

Run:  python app.py   then open http://127.0.0.1:5000
"""
import math
import os
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model (created by train_model.py)
try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as file:
        model = pickle.load(file)
except Exception:
    raise SystemExit("Could not load model.pkl. Please run:  python train_model.py")


def get_number(form, field, label, low, high, whole=False):
    """Read one input from the form and check that it is valid."""
    text = form.get(field, "").strip()
    if text == "":
        raise ValueError(f"{label} is required.")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value):          # rejects nan and inf
        raise ValueError(f"{label} must be a number.")
    if whole and value != int(value):
        raise ValueError(f"{label} must be a whole number.")
    if not (low <= value <= high):
        raise ValueError(f"{label} must be between {low} and {high}.")
    return value


def crowd_status(minutes):
    """Simple rule: decide Low / Medium / High from the predicted waiting time."""
    if minutes <= 10:
        return "Low"
    if minutes <= 30:
        return "Medium"
    return "High"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", form={})

    # 1. Validate the inputs
    try:
        people = get_number(request.form, "people", "Number of people waiting", 0, 200, whole=True)
        counters = get_number(request.form, "counters", "Number of active counters", 1, 20, whole=True)
        service = get_number(request.form, "service_time", "Average service time", 0.5, 30)
    except ValueError as error:
        return render_template("predict.html", error=str(error), form=request.form)

    # 2. Predict with the trained model
    workload = people * service / counters
    minutes = max(0, model.predict([[workload]])[0])
    status = crowd_status(minutes)

    # 3. Explain the prediction in simple words
    slope, intercept = model.coef_[0], model.intercept_
    sign = "+" if intercept >= 0 else "-"
    explanation = (
        f"{people:.0f} people x {service:g} min each = {people * service:g} minutes of total work. "
        f"This work is shared by {counters:.0f} counter(s), so each counter has about "
        f"{workload:.1f} minutes of work (this is the 'workload'). "
        f"The model learned from the dataset: waiting time = {slope:.2f} x workload {sign} {abs(intercept):.2f}. "
        f"So the estimated waiting time is about {minutes:.0f} minutes."
    )
    result = {
        "minutes": round(minutes),
        "status": status,
        "css": status.lower(),
        "explanation": explanation,
    }
    return render_template("predict.html", result=result, form=request.form)


if __name__ == "__main__":
    app.run(debug=True)
