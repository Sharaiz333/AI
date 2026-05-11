from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_knn(X_train, y_train, k=5):
    
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    return model

def save_knn(model, path="knn_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_knn(path="knn_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)