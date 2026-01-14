from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


model = joblib.load("model/fraud_job_pipeline.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        input_data = {
            'title': request.form['title'],
            'location': request.form['location'],
            'department': request.form['department'],
            'salary_range': request.form['salary_range'],
            'employment_type': request.form['employment_type'],
            'required_experience': request.form['required_experience'],
            'required_education': request.form['required_education'],
            'industry': request.form['industry'],
            'function': request.form['function'],
            'description': request.form['description'],
            'requirements': request.form['requirements'],
            'benefits': request.form['benefits'],
            'telecommuting': int(request.form['telecommuting']),
            'has_company_logo': int(request.form['has_company_logo']),
            'has_questions': int(request.form['has_questions']),
            'company_profile': request.form['company_profile']
        }

        # Engineered features (lengths)
        input_data['title_length'] = len(str(input_data['title']).split())
        input_data['description_length'] = len(str(input_data['description']).split())
        input_data['requirements_length'] = len(str(input_data['requirements']).split())

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Predict
        prob_fake = model.predict_proba(df)[:, 1][0]
        prediction = "Fake" if prob_fake >= 0.5 else "Real"

        return render_template("result.html", 
                               probability=round(prob_fake*100, 2), 
                               prediction=prediction)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
