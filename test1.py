import pandas as pd
import joblib

# Load your trained RandomForestClassifier model from the .pkl file
model = joblib.load('iris.pkl')

# Input features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
input_features = {
    'SepalLengthCm': 5.4,
    'SepalWidthCm': 3.0,
    'PetalLengthCm': 4.5,
    'PetalWidthCm': 1.5
}

# Create a DataFrame from the input data
input_data = pd.DataFrame([input_features])

# Make predictions on the input data
predicted_species_name = model.predict(input_data)[0]

print(f'Predicted Iris Species: {predicted_species_name}')
