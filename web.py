from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
# from flask_cors import CORS, cross_origin
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
# @cross_origin()
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gillcolor = float(request.form["gill-color"])
        sporeprintcolor = float(request.form["spore-print-color"])
        population = float(request.form["population"])
        gillsize = float(request.form["gill-size"])
        stalk_root = float(request.form["stalk-root"])
        bruises = float(request.form["bruises"])
        gillspacing = float(request.form["gill-spacing"])
        stalksurfaceabovering = float(request.form["stalk-surface-above-ring"])
        stalksurfacebelowring = float(request.form["stalk-surface-below-ring"])
        ringtype = float(request.form["ring-type"])
        habitat = float(request.form["habitat"])
        odor = float(request.form["odor"])

        x = pd.DataFrame({"gill-color": [gillcolor], "spore-print-color": [sporeprintcolor], "population": [population],
                          "gill-size": [gillsize], "stalk-root": [stalk_root], "bruises": [bruises],
                          "gill-spacing": [gillspacing], "stalk-surface-above-ring": [stalksurfaceabovering], 
                          "stalk-surface-below-ring": [stalksurfacebelowring] , "ring-type": [ringtype],
                          "habitat": [habitat], "odor": [odor]})
        ml = model.predict(x)
        m = round(ml[0], 2)
        if m == 0:
            g = "edible"
        else:
            g = "poisonous"
        return render_template('index.html', result='Your mushroom is {}!'.format(g))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9990)