from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from dataset import load_data
from preprocessing import preprocess_all, reshape_for_nn
from knn_model import train_knn, save_knn
from nn_model import build_model, save_model

TRAIN_PATH = "data/mnist_train.csv"
TEST_PATH  = "data/mnist_test.csv"

# Load data
X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
print("Loading MNIST dataset...")
print("(70,000 images of handwritten digits, 28x28 pixels each)\n")
print("Data loaded successfully!\n")

# Preprocess data
print("\n Performing Data Preprocessing...")
print("Normalizing pixel values (0-255) -> (0.0-1.0)...\n")
X_train, X_test = preprocess_all(X_train, X_test)

# Find Best K
print("Finding best K for KNN...")
best_k = 5
best_score = 0

for k in [3, 5, 7]:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance')
    score = cross_val_score(knn_temp, X_train[:5000], y_train[:5000], cv=3).mean()
    print(f"  k={k}  CV accuracy: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k: {best_k}\n")


# ===== Train KNN =====
print("Training KNN...")
knn = train_knn(X_train[:10000], y_train[:10000], k=best_k)
save_knn(knn)
print("\n KNN Training Completed!")
print("KNN model saved as knn_model.pkl\n")

# ===== Train Neural Network =====
print("Training Neural Network...")
X_train_nn = reshape_for_nn(X_train)
X_test_nn  = reshape_for_nn(X_test)
datagen = ImageDataGenerator(rotation_range=10,zoom_range=0.10,
width_shift_range=0.10, height_shift_range=0.10)

datagen.fit(X_train_nn)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5),
    ModelCheckpoint("nn_model.h5", save_best_only=True)
]

model = build_model()
model.fit(datagen.flow(X_train_nn, y_train, batch_size=64),
epochs=15, validation_data=(X_test_nn, y_test),
callbacks=callbacks)

save_model(model)

print("Training Completed!")