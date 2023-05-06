from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('Finalmodel (1).pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    Age = request.form.get('Age')
    Friends = request.form.get('Friends')
    Outing = request.form.get('Outing')
    Comm = request.form.get('Comm')
    Talk = request.form.get('Talk')
    Finance = request.form.get('Finance')
    Lone = request.form.get('Lone')
    Gender = request.form.get('Gender')
    Stable = request.form.get('Stable')
    Depression = request.form.get('Depression')

    input_query = np.array([[Age, Gender, Friends, Outing, Stable, Comm, Talk, Finance, Lone, Depression]])

    result = model.predict(input_query)[0]

    return jsonify({'Social Health': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
