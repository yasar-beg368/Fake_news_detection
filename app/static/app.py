from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        prediction = model.predict(vect)
        result = 'Fake News 📰❌' if prediction[0] == 'FAKE' else 'Real News ✅'
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
