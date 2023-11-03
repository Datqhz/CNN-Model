import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the CSV file
df = pd.read_csv("emotion_text.csv")
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
df['text'] = df['text'].str.lower()

# Preprocess the data
texts = df['text'].values
labels = df['label'].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)  # the number of words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# Pad sequences to a fixed length
max_seq_length = 200  # Adjust from 100 to 300. It suitable for Vietnamese language
data = pad_sequences(sequences, maxlen=max_seq_length)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=max_seq_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Evaluate the model with accuracy_score, precision, recall and f1
y_preds = model.predict(x_test)
predicted_classes = y_preds.argmax(axis=-1)
print("Classifier metrics on the test set")
print(f"Accurracy: {accuracy_score(y_test, predicted_classes)*100:.2f}%")
print(f"Precision: {precision_score(y_test, predicted_classes, average='weighted')}")
print(f"Recall: {recall_score(y_test, predicted_classes, average='weighted')}")
print(f"F1: {f1_score(y_test, predicted_classes, average='weighted')}")

# Save the tokenizer to a file
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Save the tokenizer to a file
with open('CNN_model.pkl', 'wb') as file:
    pickle.dump(model, file)

