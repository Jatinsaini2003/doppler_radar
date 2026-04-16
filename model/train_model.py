import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('../synthetic_micro_doppler_dataset.csv')

# Features and labels
X = df.iloc[:, :-1].values
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved as model.pkl")