import numpy as np
import pickle
from flask import Flask, render_template,request

app=Flask(__name__)
model=pickle.load(open('irismodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    slength=float(request.form['sepal_length'])
    swidth=float(request.form['sepal_width'])
    plength=float(request.form['petal_length'])
    pwidth=float(request.form['petal_width'])

    final_features = [np.array([slength,swidth,plength,pwidth])]
    prediction=model.predict(final_features)[0]
    return render_template('index.html', sleng=slength, pred= prediction)
    
    
    







if __name__=="__main__":
    app.run(debug=True)