from flask import Flask, render_template, request
import numpy as np
from pymongo import MongoClient
import pickle
import logging
from os import environ

#SECRET_KEY = environ.get('MONGODB_URI')
#print("SECRET_KEY: ",SECRET_KEY)
logging.basicConfig(filename='record.log', level=logging.DEBUG)
app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017")
db = client["thyroidDetection"]
collection = db["patients"]
logging.basicConfig(filename='record.log', level=logging.DEBUG)

model = pickle.load(open('models/random_forest.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        TSH = float(request.form['TSH'])
        TT4 = float(request.form['TT4'])
        FTI = float(request.form['FTI'])
        T3 = float(request.form['T3'])
        T4U = float(request.form['T4U'])
        print("T4U: ", request.form['T4U'])
        on_thyroxine = float(request.form['on_thyroxine'])
        print("on_thyroxine: ", request.form['on_thyroxine'])

        on_antithyroid_medication = float(
            request.form['on_antithyroid_medication'])
        print("on_antithyroid_medication: ", on_antithyroid_medication)
        goitre = float(request.form['goitre'])
        print("goitre: ", goitre)
        hypopituitary = float(request.form['hypopituitary'])
        print("hypopituitary: ", hypopituitary)
        psych = float(request.form['psych'])
        print("psych: ", psych)
        lithium = float(request.form['lithium'])
        print("lithium: ", lithium)
        TSH_measured = float(request.form['TSH_measured'])
        print('TSH_measured: ', TSH_measured)
        TT4_measured = float(request.form['TT4_measured'])
        print('TT4_measured: ', TT4_measured)
        T4U_measured = float(request.form['T4U_measured'])
        print("T4U_measured: ", T4U_measured)
        T3_measured = float(request.form['T3_measured'])
        print("T3_measured: ", T3_measured)
        query_on_thyroxine = float(request.form['query_on_thyroxine'])
        print("query_on_thyroxine: ", query_on_thyroxine)
        query_hyperthyroid = float(request.form['query_hyperthyroid'])
        print("query_hyperthyroid: ", query_hyperthyroid)
        query_hypothyroid = float(request.form['query_hypothyroid'])
        print("query_hypothyroid: ", query_hypothyroid)

        I131 = float(request.form['I131'])

        thyroid_surgery = float(request.form['thyroid_surgery'])
        pregnant = float(request.form['pregnant'])
        sick = float(request.form['sick'])
        tumor = float(request.form['tumor'])
        print("tumor: ", tumor)
        FTI_measured = float(request.form['FTI_measured'])
        print('FTI measured: ', FTI_measured)
        logging.debug("inputs taken")

        # Perform prediction on the patient's data and return the result
        query = np.array([
            age,
            sex,
            TSH,
            TT4,
            FTI,
            T3,
            T4U,
            on_thyroxine,
            on_antithyroid_medication,
            goitre,
            hypopituitary,
            psych,
            lithium,
            TSH_measured,
            TT4_measured,
            T4U_measured,
            T3_measured,
            query_on_thyroxine,
            query_hyperthyroid,
            query_hypothyroid,
            I131,
            thyroid_surgery,
            pregnant,
            sick,
            tumor,
            FTI_measured
        ]).reshape(1, 26)
        print("query: ",query)
        predicted_class = model.predict(query)[0]
        print("predicted class: ",predicted_class)
        if predicted_class == 'P':
            predicted_class = "Present"
        else:
            predicted_class = "Not present"
        # Create a dictionary to store the patient's data

        
        patient = {
            'age': age,
            'sex': sex,
            'TSH': TSH,
            'TT4': TT4,
            'FTI': FTI,
            'T3': T3,
            'T4U': T4U,
            'on_thyroxine': on_thyroxine,
            'on_antithyroid_medication': on_antithyroid_medication,
            'goitre': goitre,
            'hypopituitary': hypopituitary,
            'psych': psych,
            'lithium': lithium,
            'TSH_measured': TSH_measured,
            'TT4_measured': TT4_measured,
            'T4U_measured': T4U_measured,
            'T3_measured': T3_measured,
            'query_on_thyroxine': query_on_thyroxine,
            'query_hyperthyroid': query_hyperthyroid,
            'query_hypothyroid': query_hypothyroid,
            'I131': I131,
            'thyroid_surgery': thyroid_surgery,
            'pregnant': pregnant,
            'sick': sick,
            'tumor': tumor,
            'FTI_measured': FTI_measured,
            "predict":predicted_class
        }
        # Insert the patient's data into the MongoDB collection
        print("patient")
        collection.insert_one(patient)
        print("patient inserted")
        return render_template('main.html', prediction_text=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
