from flask import Flask, request, jsonify, render_template
import pickle as pickle

app = Flask(__name__)

# Load the pre-trained model
heart_model = pickle.load(open('Model/heart.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')
    

@app.route('/predict',methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    # Make a prediction using the loaded model
    prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    print(prediction)
    
    # Return the predicted value as JSON
    return render_template('predict.html', prediction =prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
