import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score
from flask import request,Flask,jsonify
import pandas as pd
app = Flask(__name__)

if not os.path.exists('model.pkl'):
    data = pd.read_csv('diabetes.csv')
    X = data.drop(['Outcome'], axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pickle.dump(model,open('model.pkl','wb'))
@app.route('/login',methods = ['GET'])
def print_hi():
    name =request.args.get("name")
    return f"hello to flask {name}"
@app.route("/predict",methods=['POST'])
def predict():
    ds = pd.DataFrame([request.json])
    model = pickle.load(open('model.pkl','rb'))
    out =model.predict(ds)
    out = True if out[0] ==1 else False
    return jsonify({"result":out})
@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(port=2000,debug=True)

