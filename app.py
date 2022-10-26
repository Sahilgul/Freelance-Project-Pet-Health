from telnetlib import SE
import numpy as np
import joblib
import pandas as pd

from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

with open('static/Xtrain' , 'rb') as i:
        X_train = pickle.load(i)
with open('static/ytrain' , 'rb') as i:
    y_train = pickle.load(i)

app = Flask(__name__, template_folder = "templates")


@app.route("//")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    FIV = request.form.get("FIV")
    Age = request.form.get("Age")
    Weight = request.form.get("Weight")
    Sex = request.form.get("Sex")
    FeLV = request.form.get ("FeLV")
    FIP = request.form.get ("FIP")
    Retrovirus = request.form.get ("Retrovirus")
    Fibre = request.form.get ("Fibre")
    Spayed = request.form.get("Spayed")
    Allergy = request.form.get ("Allergy")
    Medical_history = request.form.get("Medical_history")
    Vaccinated = request.form.get("Vaccinated")
    Pedigree = request.form.get("Pedigree")
    Toilet_visit = request.form.get("Toilet_visit")
    Housing_space = request.form.get("Housing_space")
    Outdoor = request.form.get("Outdoor")
    breed = request.form.get("breed")
    print(Age,Weight,FIV, FeLV, FIP, Retrovirus, Toilet_visit, Fibre, Spayed, Allergy, Medical_history, Vaccinated, Housing_space, Outdoor, breed)
    # randomforest = joblib.load("model")
    #randomforest = pickle.load('open('model', 'rb')')
    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=90, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
            oob_score=False, verbose=0,
            warm_start=False, random_state = 2022)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    pred_randomforest = clf.predict([[float(FIV),float(FeLV),float(FIP), float(Retrovirus), float(Toilet_visit), float(Fibre), float(Spayed), float(Allergy), float(Medical_history), float(Vaccinated), float(Housing_space), float(Pedigree), float(Outdoor), float(Sex), float(Weight), float(Age), float(breed)]])
    s_randomforest = "Not Happy! :(( " if pred_randomforest > 0 else "Happy! :))"
    s = "Your cat is " + s_randomforest
    return(s)

@app.route('/web_scrap',methods=['GET','POST'])
def web_scrap():
    return render_template('web_scr.html')


@app.route('/pet_health',methods=['GET','POST'])
def pet_health():
    return render_template('overall_health.html')

if __name__ == "__main__":
    app.run(debug=True)