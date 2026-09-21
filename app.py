import joblib
from pathlib import Path

# Create models directory if it doesn't exist
Path("models").mkdir(parents=True, exist_ok=True)

# Save your trained models (ensure variable names match your notebook)
joblib.dump(rf_model, "models/random_forest_model.pkl")      # Replace rf_model with your actual trained Random Forest variable
joblib.dump(preprocessor, "models/preprocessor.pkl")          # Replace preprocessor with your actual pipeline/scaler variable
joblib.dump(kmeans_model, "models/kmeans_model.pkl")          # Replace kmeans_model with your actual K-Means model variable

print("Model artifacts successfully saved inside the models/ directory!")
