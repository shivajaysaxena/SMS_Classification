from flask import Flask, render_template, request, jsonify
import pickle

# Load the pre-trained model and vectorizer
with open('../model/sms_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('../model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_message():
    message = request.form.get('message')
    if not message:
        return jsonify({'error': 'No message provided.'}), 400

    # Vectorize the input message
    vectorized_message = vectorizer.transform([message])

    # Predict and return result
    prediction = model.predict(vectorized_message)[0]
    label = 'Spam' if prediction == 1 else 'Not Spam'
    print('message:', message, 'classification:', label)
    return jsonify({'message': message, 'classification': label})

if __name__ == '__main__':
    app.run(debug=True)
