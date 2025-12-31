import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

"""# Create a 2D synthetic dataset for binary classification"""

X, y = make_classification(
    n_samples=250,
    n_features=2,       # keep 2 features so we can visualize
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

print("Feature shape:", X.shape)
print("Target shape :", y.shape)

plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Synthetic binary classification data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

"""#Training SVM Model"""

svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)

print("Training done.")

"""#Evaluation"""

y_pred = svm_clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""#PLOT"""

# Create a mesh grid over feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Predict over the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = svm_clf.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 5))
# Decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")

# Training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

plt.title("SVM decision boundary (linear kernel)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()