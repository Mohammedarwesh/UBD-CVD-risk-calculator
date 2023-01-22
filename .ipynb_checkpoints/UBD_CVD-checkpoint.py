import numpy as np
from flask import Flask, request,app,url_for,jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
## load
model = pickle.load(open('validated_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform(np.array(data).reshape(1,-1))
    print(scaled_data)
    output = model.predict(scaled_data)
    risk = output
    msg = "Low CVD risk" if risk == 0 else "High CVD risk"
    probapility = model.predict_proba(scaled_data)
    probapility_round = round(probapility[0,1]*100)


    return render_template('predict.html', 
                           prediction_text=f'You are of {msg} and Your CVD risk is about {probapility_round}%')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)