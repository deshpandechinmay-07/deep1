from flask import Flask, request, render_template
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# --------------------------------
# Load the NEW PKL bundle
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "fertility_deep_learning_bundle.pkl"
)

with open(MODEL_PATH, "rb") as file:
    bundle = pickle.load(file)

model = bundle["model"]
numeric_pipeline = bundle["numeric_pipeline"]
categorical_pipeline = bundle["categorical_pipeline"]

numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
feature_columns = bundle["feature_columns"]

print("--------------------------------")
print("Model loaded successfully")
print("Model type:", bundle["model_type"])
print("Expected features:", bundle["input_dim"])
print("--------------------------------")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------
        # Read form inputs
        # --------------------------------
        user_data = {
            "Season": request.form["Season"],

            "Age": float(
                request.form["Age"]
            ),

            "Childish diseases": request.form[
                "Childish diseases"
            ],

            "Accident or serious trauma": request.form[
                "Accident or serious trauma"
            ],

            "Surgical intervention": request.form[
                "Surgical intervention"
            ],

            "High fevers in the last year": request.form[
                "High fevers in the last year"
            ],

            "Frequency of alcohol consumption": request.form[
                "Frequency of alcohol consumption"
            ],

            "Smoking habit": request.form[
                "Smoking habit"
            ],

            "Number of hours spent sitting per day": float(
                request.form[
                    "Number of hours spent sitting per day"
                ]
            )
        }

        input_df = pd.DataFrame([user_data])

        # --------------------------------
        # Numerical preprocessing
        # --------------------------------
        input_num = input_df[numeric_cols]

        input_num = numeric_pipeline.transform(
            input_num
        )

        # --------------------------------
        # Categorical preprocessing
        # --------------------------------
        input_cat = input_df[categorical_cols]

        input_cat = categorical_pipeline.transform(
            input_cat
        )

        # --------------------------------
        # Combine
        # --------------------------------
        final_features = np.concatenate(
            [input_num, input_cat],
            axis=1
        )

        # --------------------------------
        # Safety check
        # --------------------------------
        if final_features.shape[1] != len(feature_columns):
            raise ValueError(
                "Feature mismatch: model expects "
                + str(len(feature_columns))
                + " features, but received "
                + str(final_features.shape[1])
            )

        print(
            "Prediction input shape:",
            final_features.shape
        )

        # --------------------------------
        # Prediction
        # --------------------------------
        prediction = model.predict(
            final_features
        )[0]

        probability = model.predict_proba(
            final_features
        )[0][1]

        if prediction == 1:
            output = "Altered"
        else:
            output = "Normal"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {output}",
            probability=f"{probability * 100:.2f}%"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
