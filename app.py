from flask import Flask, render_template, request
import pickle
import numpy as np
from database import *
from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__,static_url_path='/static')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
# Load the machine learning model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib 
# 
  
@app.route('/p')
def p():
    return render_template('index.html')

@app.route('/')
def m():
    return render_template('main.html')

@app.route('/l')
def l():
    return render_template('login.html')

@app.route('/h')
def h():
    return render_template('home.html')

@app.route('/r')
def r():
    return render_template('register.html')

@app.route('/m')
def menu():
    return render_template('menu.html')



@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/home.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")

from flask import Flask, render_template, request
import pandas as pd
import pickle

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs from HTML
    device_id = request.form['device_id']  # Optional; not used in prediction
    iomt_type = request.form['iomt_type']
    location = request.form['location']
    data_transferred = float(request.form['data_transferred'])  # in MB
    access_time = float(request.form['access_time'])            # in ms
    auth_method = request.form['auth_method']
    requests_per_minute = int(request.form['requests_per_minute'])
    data_format = request.form['data_format']
    os_type = request.form['os_type']
    alert_level = request.form['alert_level']

    # Create input dictionary (excluding device_id)
    input_data = {
        'IoMT_Type': iomt_type,
        'Location': location,
        'Data_Transferred_MB': data_transferred,
        'Access_Time(ms)': access_time,
        'Auth_Method': auth_method,
        'Requests_Per_Minute': requests_per_minute,
        'Data_Format': data_format,
        'OS_Type': os_type,
        'Alert_Level': alert_level
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Load model and encoders
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    le_attack = joblib.load('attack_label_encoder.pkl')

    # Preprocess categorical columns
    categorical_cols = ['IoMT_Type', 'Location', 'Auth_Method', 'Data_Format', 'OS_Type', 'Alert_Level']
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale the numerical features
    input_scaled = scaler.transform(input_df)

    # Predict using the trained model
    prediction = best_model.predict(input_scaled)

    # Decode prediction
    predicted_attack = le_attack.inverse_transform(prediction)[0]

    # Render result to template
    return render_template("result.html", op1=f"Predicted Attack Type: {predicted_attack}")



if __name__ == "__main__":
    app.run(debug=True, port=5112)