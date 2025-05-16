from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form and convert to correct types
        department = int(request.form['department'])
        region = int(request.form['region'])
        education = int(request.form['education'])
        gender = int(request.form['gender'])
        recruitment_channel = int(request.form['recruitment_channel'])
        no_of_trainings = float(request.form['no_of_trainings'])
        age = float(request.form['age'])
        previous_year_rating = float(request.form['previous_year_rating'])
        length_of_service = float(request.form['length_of_service'])
        awards_won = float(request.form['awards_won'])
        avg_training_score = float(request.form['avg_training_score'])

        # Prepare input features for prediction (ensure order matches training data)
        input_features = np.array([[department, region, education, gender, recruitment_channel,
                                    no_of_trainings, age, previous_year_rating, length_of_service,
                                    awards_won, avg_training_score]])

        # Predict promotion (0 or 1)
        prediction = model.predict(input_features)[0]

        # Decide the message based on prediction
        if prediction == 1:
            result = "Employee will be PROMOTED 🎉"
        else:
            result = "Employee will NOT be promoted."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # For debugging, print error and show friendly message
        print("Error:", e)
        return render_template('index.html', prediction_text="Something went wrong. Please check input values.")

if __name__ == '__main__':
    app.run(debug=True)
