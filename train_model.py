from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model Saved Successfully!")