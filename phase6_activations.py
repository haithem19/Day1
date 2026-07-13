import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

activations = ['sigmoid', 'tanh', 'relu']
results = []
histories = {}


def first_epoch_below_threshold(values, threshold=0.1):
    for idx, val in enumerate(values, start=1):
        if val < threshold:
            return idx
    return 'N/A'


def train_activation_case(name, layer_dims_activations):
    tf.random.set_seed(42)

    layers = [
        keras.layers.Dense(units, activation=act, input_shape=(784,) if i == 0 else None)
        for i, (units, act) in enumerate(layer_dims_activations)
    ]
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_losses = history.history['val_loss']
    convergence_epoch = first_epoch_below_threshold(val_losses, threshold=0.1)

    results.append({
        'activation': name,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'convergence_epoch_sub01': convergence_epoch,
        'train_time_s': elapsed,
    })
    histories[name] = val_losses


for activation in activations:
    train_activation_case(activation, [(128, activation), (64, activation), (10, 'softmax')])

# Cas limite : couche dense linéaire (sans activation explicite)
train_activation_case('linear', [(128, None), (64, None), (10, 'softmax')])

# Scénario adversarial : softmax aussi dans les couches cachées
train_activation_case('softmax_hidden', [(128, 'softmax'), (64, 'softmax'), (10, 'softmax')])

# Pour aller plus loin : impact de la profondeur avec ReLU (meilleure activation)
train_activation_case('relu_shallow', [(256, 'relu'), (10, 'softmax')])
train_activation_case('relu_medium', [(128, 'relu'), (64, 'relu'), (10, 'softmax')])
train_activation_case('relu_deep', [(128, 'relu'), (64, 'relu'), (32, 'relu'), (10, 'softmax')])

print('\n=== TABLEAU COMPARATIF ===')
print(f"{'Activation':10s} | {'Val loss epoch 10':18s} | {'Test accuracy':14s} | {'Epoch < 0.1 loss':16s} | {'Temps (s)':10s}")
print('-' * 95)
for r in results:
    print(
        f"{r['activation']:10s} | "
        f"{r['val_loss_final']:.4f}{'':14s} | "
        f"{r['test_accuracy']:.4f}{'':9s} | "
        f"{str(r['convergence_epoch_sub01']):16s} | "
        f"{r['train_time_s']:.0f}"
    )

plt.figure(figsize=(10, 5))
for activation, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=activation, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title("Convergence selon la fonction d'activation (MNIST)")
plt.legend()
plt.savefig('phase6_activations_curve.png', dpi=100, bbox_inches='tight')

print('\nCourbe sauvegardée : phase6_activations_curve.png')
