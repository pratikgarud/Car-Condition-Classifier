from flask import Flask,render_template,request,redirect
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('CarClassifier.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifier',methods=['GET','POST'])
def classifier():
    if request.method == 'POST':
        price = request.form.get('carprice')
        door = request.form.get('doors')
        maint = request.form.get('main')
        person = request.form.get('person')
        lugg = request.form.get('luggage')
        safe = request.form.get('safety')
        feat = np.array([price,door,maint,person,lugg,safe])
        pre = model.predict([feat])
        pred = np.asscalar(pre)
    return render_template('index.html', output=pred)

if __name__=='__main__':
    app.run(debug=True)