Fake News Detection using NLP

Detect fake news articles using Machine Learning, NLP, and Flask.

📌 Project Overview

Fake news spreads rapidly across social media and online platforms. This project builds an NLP-based classifier that predicts whether a news article is REAL or FAKE.
The model uses TF-IDF for feature extraction and Naive Bayes for classification, achieving an F1-score of 0.89.
A simple Flask web app is included for real-time predictions.

🚀 Features

Cleaned and processed text using NLP

TF-IDF vectorization for feature extraction

Naive Bayes classifier

92% model accuracy

Flask web app for real-time fake news checking

Easy to run locally

📂 Project Structure
fake_news_detection/
│
├── dataset/
│   └── fake_or_real_news.csv
│
├── app/
│   ├── static/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│
├── model/
│   ├── fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── notebook/
│   └── fake_news_detection.ipynb
│
└── requirements.txt

🧠 Tech Stack

Python

Flask

Scikit-learn

NLTK

NumPy & Pandas

Bootstrap (for UI)

🧹 Data Preprocessing

✔ Convert text to lowercase
✔ Remove punctuation, numbers, URLs
✔ Remove stopwords
✔ Clean HTML tags
✔ Tokenization

🔥 Model Used

TF-IDF Vectorizer

Multinomial Naive Bayes Classifier

📈 Model Performance
Metric	Score
Accuracy	92%
Precision	0.90
Recall	0.89
F1-score	0.89
🖥️ Web App Screenshot

(Add screenshot here)

![App UI](app_screenshot.png)

🛠️ Installation & Setup
1️⃣ Clone the Repo
git clone https://github.com/yourusername/fake_news_detection.git
cd fake_news_detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Flask App
cd app
python app.py


Open in your browser:

http://127.0.0.1:5000/

🧪 How to Use

Enter a news article or paragraph in the text box

Click on Check

The model predicts:

Real News ✅

Fake News ❌

📦 Model Files
File	Description
fake_news_model.pkl	Trained Naive Bayes model
tfidf_vectorizer.pkl	TF-IDF fitted vectorizer
📘 Jupyter Notebook

Full training workflow is available at:

notebook/fake_news_detection.ipynb

🚀 Future Enhancements

Deploy on cloud platforms (Render, Railway, Heroku)

Add deep learning models (LSTM, BERT)

Create a REST API endpoint

Add more datasets for robustness

🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss what you'd like to improve.