import pandas as pd
import re
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.debug = True
CORS(app)

df_emotion = pd.read_csv('emotion_text.csv')
def save_new_assess(text, mylabel):
    global df_emotion
    new_row = pd.DataFrame({'text': [text], 'label': [mylabel]})
    # Concatenate the original DataFrame with the new row
    df_emotion = pd.concat([df_emotion, new_row], ignore_index=True)
    df_emotion.to_csv('emotion_text.csv', index = False)

@app.route('/api/save-assess', methods=['POST'])
def save_assess():
    try:
        request_data = request.get_json()
        if 'text'  in request_data:
            text = request_data['text']
            save_new_assess(text,request_data['label'])
            response = {
                "message": "Data added successfully",
            }
            return jsonify(response)
    except Exception as e:
        return jsonify(error=str(e))


@app.route('/api/predict/svm', methods=['POST'])
def predict_emotion_SVC():
    try:
        request_data = request.get_json()
        print(request_data)
        if 'text'  in request_data:
            text = request_data['text']
            with open('tfidf_vectorize.pkl', 'rb') as f:
                loaded_vectorizer = pickle.load(f)
            with open('linearSVC_model.pkl', 'rb') as f:
                model = pickle.load(f)
            text_vectorize = loaded_vectorizer.transform([text])
            predict = model.predict(text_vectorize)
            response  = {
                "message": "Data added successfully",
                "data": predict.tolist()
            }
            return jsonify(response)
    except Exception as e:
        return jsonify(error=str(e))
@app.route('/api/predict', methods=['POST'])
def predict_emotion_CNN():
    try:
        request_data = request.get_json()
        print(request_data)
        if 'text' in request_data:
            text = request_data['text']
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            with open('CNN_model.pkl', 'rb') as f:
                model = pickle.load(f)
            text = re.sub(r'\s+', ' ', text).strip().lower()
            new_sequences = tokenizer.texts_to_sequences([text])
            new_data = pad_sequences(new_sequences, maxlen=200)
            predict = model.predict(new_data)
            predicted_classes = predict.argmax(axis=-1)
            response  = {
                "message": "Data added successfully",
                "data": predicted_classes.tolist()
            }
            return jsonify(response)
    except Exception as e:
        return jsonify(error=str(e))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(port=5000)



