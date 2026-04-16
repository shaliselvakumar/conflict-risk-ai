from model import build_model

def train_model(X, y):

    model = build_model()
    model.fit(X, y)

    return model