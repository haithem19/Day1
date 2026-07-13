import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def init_he(in_dim, out_dim, seed):
    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / in_dim)
    return rng.standard_normal((in_dim, out_dim)) * std, np.zeros(out_dim)


def train_numpy_network(X_train, y_train, X_test, y_test, n_epochs=200, lr=0.01):
    W1, b1 = init_he(X_train.shape[1], 16, 43)
    W2, b2 = init_he(16, 8, 44)
    W3, b3 = init_he(8, 1, 45)

    losses = []
    for epoch in range(n_epochs):
        z1 = X_train @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        y_pred = sigmoid(z3).flatten()

        loss = bce_loss(y_train, y_pred)
        losses.append(loss)

        err3 = y_pred - y_train
        dW3 = (a2.T @ err3.reshape(-1, 1)) / len(X_train)
        db3 = np.array([np.mean(err3)])

        err2 = (err3.reshape(-1, 1) @ W3.T) * relu_grad(z2)
        dW2 = (a1.T @ err2) / len(X_train)
        db2 = np.mean(err2, axis=0)

        err1 = (err2 @ W2.T) * relu_grad(z1)
        dW1 = (X_train.T @ err1) / len(X_train)
        db1 = np.mean(err1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

    test_pred = sigmoid(relu(relu(X_test @ W1 + b1) @ W2 + b2) @ W3 + b3).flatten()
    test_acc = np.mean((test_pred > 0.5) == y_test)

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3,
    }
    return losses, test_acc, params


def numpy_predict(params, X):
    z1 = X @ params['W1'] + params['b1']
    a1 = relu(z1)
    z2 = a1 @ params['W2'] + params['b2']
    a2 = relu(z2)
    z3 = a2 @ params['W3'] + params['b3']
    return sigmoid(z3).flatten()


def build_keras_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_keras_network(X_train, y_train, X_test, y_test):
    tf.random.set_seed(42)
    model = build_keras_model(X_train.shape[1])
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )
    elapsed = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return model, history, elapsed, test_loss, test_acc


def evaluate_missing_values_effect(X_train, y_train, X_test, y_test):
    X_train_bad = X_train.copy()
    X_train_bad[: int(0.1 * len(X_train_bad)), 0] = 0.0

    model_bad = build_keras_model(X_train.shape[1])
    model_bad.fit(X_train_bad, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    _, acc_bad = model_bad.evaluate(X_test, y_test, verbose=0)

    X_train_fixed = X_train_bad.copy()
    col = 0
    non_zero = X_train_fixed[:, col] != 0
    median = np.median(X_train_fixed[non_zero, col])
    X_train_fixed[:, col] = np.where(X_train_fixed[:, col] == 0, median, X_train_fixed[:, col])

    model_fixed = build_keras_model(X_train.shape[1])
    model_fixed.fit(X_train_fixed, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    _, acc_fixed = model_fixed.evaluate(X_test, y_test, verbose=0)

    return acc_bad, acc_fixed


def evaluate_normalization_effect(X_train_raw, y_train, X_test_raw, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    model_scaled = build_keras_model(X_train_raw.shape[1])
    history_scaled = model_scaled.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0
    )
    _, acc_scaled = model_scaled.evaluate(X_test_scaled, y_test, verbose=0)

    model_raw = build_keras_model(X_train_raw.shape[1])
    history_raw = model_raw.fit(
        X_train_raw, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0
    )
    _, acc_raw = model_raw.evaluate(X_test_raw, y_test, verbose=0)

    return history_scaled, acc_scaled, history_raw, acc_raw


def evaluate_optimizer_comparison(X_train, y_train, X_test, y_test):
    optimizers_to_test = [
        ('Adam lr=1e-3', keras.optimizers.Adam(learning_rate=1e-3)),
        ('SGD lr=1e-3', keras.optimizers.SGD(learning_rate=1e-3)),
        ('SGD lr=1e-2', keras.optimizers.SGD(learning_rate=1e-2)),
    ]
    comparison = []
    for label, optimizer in optimizers_to_test:
        tf.random.set_seed(42)
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        comparison.append({'label': label, 'test_accuracy': acc})
    return comparison


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Shape X : {X.shape} | Classes : {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    numpy_losses, numpy_acc, numpy_params = train_numpy_network(
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs=200,
        lr=0.01,
    )

    keras_model, keras_history, keras_time, keras_test_loss, keras_test_acc = train_keras_network(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print(f"Numpy from-scratch | Loss finale : {numpy_losses[-1]:.4f} | Test accuracy : {numpy_acc:.4f}")
    print(f"Keras              | Loss finale : {keras_test_loss:.4f} | Test accuracy : {keras_test_acc:.4f}")
    print(f"Gain Keras vs Numpy : {(keras_test_acc - numpy_acc) * 100:.2f} points de %")

    acc_bad, acc_fixed = evaluate_missing_values_effect(X_train, y_train, X_test, y_test)
    print(f"\nCas limite valeurs manquantes simulées (colonne 0 mise à 0):")
    print(f"Sans correction médiane : {acc_bad:.4f} | Avec correction médiane : {acc_fixed:.4f}")

    X_extreme = np.array([[99999.0] * X_train.shape[1]])
    X_extreme_scaled = scaler.transform(X_extreme)
    pred_numpy = numpy_predict(numpy_params, X_extreme_scaled)[0]
    pred_keras = keras_model.predict(X_extreme_scaled, verbose=0).flatten()[0]
    print('\nScénario adversarial OOD (valeurs extrêmes):')
    print(f"Prédiction numpy : {pred_numpy:.6f}")
    print(f"Prédiction keras : {pred_keras:.6f}")

    # Pour aller plus loin : impact de la normalisation
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _, acc_scaled, _, acc_raw = evaluate_normalization_effect(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    print('\nImpact de la normalisation (StandardScaler) :')
    print(f"Avec normalisation : {acc_scaled:.4f} | Sans normalisation : {acc_raw:.4f}")

    # Pour aller plus loin : SGD vs Adam
    optimizer_comparison = evaluate_optimizer_comparison(X_train, y_train, X_test, y_test)
    print('\nSGD vs Adam (test accuracy) :')
    for entry in optimizer_comparison:
        print(f"{entry['label']:16s} | {entry['test_accuracy']:.4f}")

    # Pour aller plus loin : facteur de vitesse numpy vs keras (par epoch)
    numpy_time_per_epoch = None
    start_numpy_timed = time.time()
    train_numpy_network(X_train, y_train, X_test, y_test, n_epochs=200, lr=0.01)
    numpy_total_time = time.time() - start_numpy_timed
    numpy_time_per_epoch = numpy_total_time / 200
    keras_time_per_epoch = keras_time / 50
    print(f"\nTemps/epoch numpy : {numpy_time_per_epoch * 1000:.2f} ms | Temps/epoch keras : {keras_time_per_epoch * 1000:.2f} ms")
    print(f"Facteur de vitesse Keras vs Numpy : x{numpy_time_per_epoch / keras_time_per_epoch:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(numpy_losses, label='numpy train loss')
    axes[0].plot(keras_history.history['loss'], label='keras train loss')
    axes[0].plot(keras_history.history['val_loss'], label='keras val loss')
    axes[0].set_title('Loss comparaison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].bar(['Numpy', 'Keras'], [numpy_acc, keras_test_acc], color=['#e76f51', '#2a9d8f'])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title('Test accuracy comparée')

    plt.savefig('phase8_comparaison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Temps d'entraînement Keras : {keras_time:.2f}s")
    print('Comparaison sauvegardée : phase8_comparaison.png')


if __name__ == '__main__':
    main()
