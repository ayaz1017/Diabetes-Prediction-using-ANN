from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("diabetes_model.h5")

def predict_diabetes(features):
    features_array = np.array([features]).astype(float)  # Convert input to NumPy array
    prediction = model.predict(features_array)[0][0]  # Get the prediction
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        glucose = float(request.form['glucose'])

        input_data = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]

        prediction = predict_diabetes(input_data)

        result_text = "The model predicts a high likelihood of diabetes." if prediction > 0.5 else "The model predicts a low likelihood of diabetes."

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

