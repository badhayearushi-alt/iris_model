 from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# Train SVM model
model = SVC(C=20, kernel="linear")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "best_model.pkl")

print("✅ best_model.pkl created successfully!")