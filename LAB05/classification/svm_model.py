from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train, kernel='linear', **kwargs):
    model = SVC(kernel=kernel, **kwargs)
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)