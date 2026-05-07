from dataset import load_data
from preprocessing import preprocess_all, reshape_for_nn
from knn_model import load_knn
from nn_model import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

print("=" * 50)
print("  Evaluating Models...")
print("=" * 50 + "\n")

TRAIN_PATH = "data/mnist_train.csv"
TEST_PATH  = "data/mnist_test.csv"

X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
X_train, X_test = preprocess_all(X_train, X_test)

# KNN
knn = load_knn()
knn_acc = knn.score(X_test[:2000], y_test[:2000])
print("KNN Accuracy:", knn_acc)
print(f"  KNN Accuracy      : {knn_acc * 100:.2f}%")

# NN
X_test_nn = reshape_for_nn(X_test)
model = load_model()
loss, acc = model.evaluate(X_test_nn, y_test)
print("Neural Network Accuracy:", acc)
print(f"  Neural Net Loss   : {loss:.4f}")
print(f"  Neural Net Accuracy: {acc * 100:.2f}%\n")

# ===== CONFUSION MATRIX =====
# KNN Predictions
y_pred_knn = knn.predict(X_test[:2000])

cm = confusion_matrix(y_test[:2000], y_pred_knn)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# ===== SAMPLE PREDICTIONS =====
# Show some test images with predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(28, 28)
    pred = y_pred_knn[i]

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {pred}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()

print("\n  Evaluation Complete!")
print("  Saved: confusion_matrix.png")
print("  Saved: sample_predictions.png")