from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

# Create models
svm_model = SVC()
rf_model = RandomForestClassifier()

# Train models
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save models
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(rf_model, "models/rf_model.pkl")

print("Models trained and saved!")