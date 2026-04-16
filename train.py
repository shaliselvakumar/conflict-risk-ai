from model import build_model

def train_model(X, y):

    model = build_model()
    model.fit(X, y, epochs=30, batch_size=4, verbose=0)

    return model