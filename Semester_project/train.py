from dataset import load_data
from preprocessing import preprocess_all, reshape_for_nn
from knn_model import train_knn, save_knn
from nn_model import build_model, save_model

TRAIN_PATH = "data/mnist_train.csv"
TEST_PATH  = "data/mnist_test.csv"

# Load data
X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)

# Preprocess
X_train, X_test = preprocess_all(X_train, X_test)

# ===== Train KNN =====
print("Training KNN...")
knn = train_knn(X_train[:10000], y_train[:10000])
save_knn(knn)

# ===== Train Neural Network =====
print("Training Neural Network...")
X_train_nn = reshape_for_nn(X_train)
X_test_nn  = reshape_for_nn(X_test)

model = build_model()
model.fit(X_train_nn, y_train, epochs=5, batch_size=64)

save_model(model)

print("Training Completed!")