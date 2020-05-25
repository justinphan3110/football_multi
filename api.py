import flask
from flask import request, jsonify
from pyvi import ViTokenizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

app = flask.Flask(__name__)
app.config['DEBUG'] = True


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set()
        with open("data/vietnamese-stopwords.txt", 'r', encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line.strip())

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = [ViTokenizer.tokenize(text.lower()) for text in X]
        return [" ".join([token for token in text.split() if token not in self.stopwords]) for text in result]


lb = joblib.load("models/football_labelencoding.pkl")
classifier = joblib.load("models/football_multilabel.pkl")


@app.route('/api/predict_football_predicate', methods=['POST'])
def predict_predicate():
    if 'text' in request.args:
        text = request.args['text']
        print("Input text:", text)
        predicted = classifier.predict([text, ])
        label = lb.inverse_transform(predicted)
        print("Label:", label)
        return jsonify({'text' : text,'predicate' : label})


if __name__ == '__main__':
    app.run()

