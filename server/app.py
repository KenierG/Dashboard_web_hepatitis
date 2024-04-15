from flask import Flask, request
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL = joblib.load('model_linear_regression.pkl')
MODEL_SEX = joblib.load('multi_output_classfifier_sex.pkl')
MODEL_AGE = joblib.load('Random_forest_classifier_age_category.pkl')


    
@app.route('/', methods=['POST'])
def hello():       
    data = request.json 

    data_list = list(data.values())

    x0 = np.array(data_list, dtype=float)
    
    prediction = MODEL.predict(x0.reshape(1, -1))  

    prediction_list = prediction.tolist()

    arr={
        'fatigue':prediction_list[0][0],
        'malaise':prediction_list[0][1],
        'anorexia':prediction_list[0][2],
        'liver_big':prediction_list[0][3],
        'liver_firm':prediction_list[0][4],
        'spleen_palable':prediction_list[0][5],
        'spiders':prediction_list[0][6],
        'ascites':prediction_list[0][7],
        'varices':prediction_list[0][8],
        'histology':prediction_list[0][9]
    }
    
    data_list2 = list(arr.values())
    nuevos_datos_array = np.array(data_list2)

    x1 = np.concatenate((x0, nuevos_datos_array), axis=0)


    probability = MODEL_SEX.predict_proba(x1.reshape(1, -1))
    prediction2 = MODEL_SEX.predict(x1.reshape(1, -1))

    arr2={
        'sex':prediction2[0][0]
    }

    data_list3 = list(arr2.values())
    nuevos_datos_array2 = np.array(data_list3)

    x2 = np.concatenate((x1, nuevos_datos_array2), axis=0)

    probability_age = MODEL_AGE.predict_proba(x2.reshape(1, -1))

    return {'number': prediction_list,'sex_pro':probability[0].tolist(),'sex':prediction2.tolist(),'age':probability_age.tolist()}

if __name__ == "__main__":
    app.run(debug=True)