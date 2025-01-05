from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load your pre-trained model (assumes you have already saved the model as 'model.pkl')
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Function to preprocess data for the model
def preprocess_input(data):
    df = pd.DataFrame([data])
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Time'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
    df['Time'] = df['Time'].astype('int64') // 10**9
    df['Date'] = df['Date'].astype('int64') // 10**9
    df = pd.get_dummies(df, columns=['Payment_type', 'Sender_bank_location', 'Receiver_bank_location', 
                                      'Payment_currency', 'Received_currency'])
    return df

# Align the input with the model features
def align_input(input_df, train_columns):
    return input_df.reindex(columns=train_columns, fill_value=0)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = request.form.to_dict()
    input_data['Amount'] = float(input_data['Amount'])  # Ensure numeric format
    
    # Preprocess the data
    processed_data = preprocess_input(input_data)
    aligned_data = align_input(processed_data, model.feature_names_in_)  # Align with trained columns

    # Perform prediction
    prediction = model.predict(aligned_data)[0]
    result = "Suspicious" if prediction == 1 else "Not Suspicious"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
