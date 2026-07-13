import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


tf.keras.utils.set_random_seed(42)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Préprocessing : flatten 28x28 → 784, normaliser entre 0 et 1
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}")


def build_model():
    return keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


def train_and_evaluate(epochs, batch_size, validation_split=0.1, verbose=1):
    model = build_model()
    compile_model(model)
    model.summary()
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    elapsed = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return model, history, elapsed, test_loss, test_acc


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.savefig("phase5_mnist_curves.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("Courbes sauvegardées : phase5_mnist_curves.png")


def check_epochs_zero():
    print("\nCas limite : epochs=0")
    model = build_model()
    compile_model(model)
    try:
        history = model.fit(
            X_train,
            y_train,
            epochs=0,
            batch_size=64,
            validation_split=0.1,
            verbose=0,
        )
        print(f"Historique vide : {len(history.history)} clés | contenu : {history.history}")
    except Exception as exc:
        print(f"Erreur Keras sur epochs=0 : {type(exc).__name__}: {exc}")


def compare_batch_sizes():
    print("\nComparaison batch_size=64 vs batch_size=1 sur 1 epoch")
    _, history_64, elapsed_64, test_loss_64, test_acc_64 = train_and_evaluate(
        epochs=1,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )
    _, history_1, elapsed_1, test_loss_1, test_acc_1 = train_and_evaluate(
        epochs=1,
        batch_size=1,
        validation_split=0.1,
        verbose=0,
    )
    print(
        f"batch_size=64 | temps: {elapsed_64:.1f}s | loss: {test_loss_64:.4f} | acc: {test_acc_64:.4f}"
    )
    print(
        f"batch_size=1  | temps: {elapsed_1:.1f}s | loss: {test_loss_1:.4f} | acc: {test_acc_1:.4f}"
    )
    print(
        f"Loss train batch_size=64: {history_64.history['loss'][-1]:.4f} | batch_size=1: {history_1.history['loss'][-1]:.4f}"
    )


def main():
    model, history, elapsed, test_loss, test_acc = train_and_evaluate(
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    print(f"\nTemps d'entraînement : {elapsed:.1f}s")
    print(f"Test accuracy        : {test_acc:.4f}")
    print(f"Test loss            : {test_loss:.4f}")
    plot_history(history)

    check_epochs_zero()
    compare_batch_sizes()


if __name__ == "__main__":
    main()