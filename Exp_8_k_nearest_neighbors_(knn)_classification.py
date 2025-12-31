from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

print("KNN model trained successfully!")

"""#Plot the initial cluster"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
plt.figure(figsize=(4,4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Initial Clusters with True Labels')
plt.show()

"""# Make predictions on the test set"""

y_pred = knn.predict(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2f}\n")

"""# Display correct and wrong predictions"""

print("Correct Predictions:")
correct_indices = np.where(y_pred == y_test)
for i in correct_indices[0]:
    print(f"  Actual: {iris.target_names[y_test[i]]}, Predicted: {iris.target_names[y_pred[i]]}, Features: {X_test[i]}")

print("\nWrong Predictions:")
wrong_indices = np.where(y_pred != y_test)
if len(wrong_indices[0]) == 0:
    print("  No wrong predictions found. The model achieved 100% accuracy on the test set.")
else:
    for i in wrong_indices[0]:
        print(f"  Actual: {iris.target_names[y_test[i]]}, Predicted: {iris.target_names[y_pred[i]]}, Features: {X_test[i]}")