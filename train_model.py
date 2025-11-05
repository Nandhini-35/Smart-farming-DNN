import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ✅ Load dataset locally (make sure the CSV file is in the same folder)
df = pd.read_csv("Crop_recommendation.csv")

# Split data into features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("smart_farming_model.pkl", "wb"))

print("✅ Model trained and saved successfully as 'smart_farming_model.pkl'")

