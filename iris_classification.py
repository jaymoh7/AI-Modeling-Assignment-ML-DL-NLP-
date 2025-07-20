# iris_classification.py

# 1. Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 2. Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Optional: Convert target to string labels for encoding demo
target_names = iris.target_names
y = y.apply(lambda i: target_names[i])

# 3. Check for missing values
print("Missing values:\n", X.isnull().sum())

# 4. Encode labels (from species names to numeric values)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
recall = recall_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
