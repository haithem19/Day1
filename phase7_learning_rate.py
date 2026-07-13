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

def first_epoch_below_threshold_lr(values, threshold=0.1):
    for idx, val in enumerate(values, start=1):
        if val < threshold:
            return idx
    return 'N/A'


learning_rates = [1e-7, 1e-3, 1.0]
lr_labels = ['trop petit (1e-7)', 'sweet spot (1e-3)', 'trop grand (1.0)']
results = []
histories = {}

for lr, label in zip(learning_rates, lr_labels):
    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
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
    histories[label] = val_losses

    results.append({
        'lr': lr,
        'label': label,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'train_time_s': elapsed,
        'delta_loss_epoch1_to10': val_losses[0] - val_losses[-1],
    })

print('\n=== TABLEAU COMPARATIF LEARNING RATE ===')
print(f"{'LR':8s} | {'Label':24s} | {'Val loss final':14s} | {'Test acc':10s} | {'Temps (s)':10s} | {'Delta loss e1->e10':18s}")
print('-' * 120)
for r in results:
    print(
        f"{r['lr']:.0e}    | "
        f"{r['label']:24s} | "
        f"{r['val_loss_final']:.4f}{'':9s} | "
        f"{r['test_accuracy']:.4f}{'':6s} | "
        f"{r['train_time_s']:.0f}{'':7s} | "
        f"{r['delta_loss_epoch1_to10']:.4f}"
    )

plt.figure(figsize=(10, 5))
for label, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title('Impact du learning rate sur la convergence (MNIST)')
plt.legend()
plt.yscale('log')
plt.savefig('phase7_lr_curve.png', dpi=100, bbox_inches='tight')

print('\nCourbe sauvegardée : phase7_lr_curve.png')

# Pour aller plus loin : SGD vs Adam au sweet spot
optimizers_to_test = [
    ('Adam lr=1e-3', keras.optimizers.Adam(learning_rate=1e-3)),
    ('SGD lr=1e-3', keras.optimizers.SGD(learning_rate=1e-3)),
    ('SGD lr=1e-2', keras.optimizers.SGD(learning_rate=1e-2)),
]
optimizer_results = []
optimizer_histories = {}

for label, optimizer in optimizers_to_test:
    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=optimizer,
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
    convergence_epoch = first_epoch_below_threshold_lr(val_losses, threshold=0.1)
    optimizer_histories[label] = val_losses

    optimizer_results.append({
        'label': label,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'train_time_s': elapsed,
        'convergence_epoch': convergence_epoch,
    })

print('\n=== TABLEAU COMPARATIF OPTIMIZER (SGD vs Adam) ===')
print(f"{'Optimizer':16s} | {'Val loss final':14s} | {'Test acc':10s} | {'Epoch < 0.1':12s} | {'Temps (s)':10s}")
print('-' * 80)
for r in optimizer_results:
    print(
        f"{r['label']:16s} | "
        f"{r['val_loss_final']:.4f}{'':9s} | "
        f"{r['test_accuracy']:.4f}{'':6s} | "
        f"{str(r['convergence_epoch']):12s} | "
        f"{r['train_time_s']:.0f}"
    )

plt.figure(figsize=(10, 5))
for label, val_losses in optimizer_histories.items():
    plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title('SGD vs Adam au sweet spot (MNIST)')
plt.legend()
plt.savefig('phase7_optimizer_curve.png', dpi=100, bbox_inches='tight')

print('\nCourbe sauvegardée : phase7_optimizer_curve.png')
