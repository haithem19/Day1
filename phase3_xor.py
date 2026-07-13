import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def initialize_parameters(hidden_units=2, seed=42, scale=0.5):
    np.random.seed(seed)
    W1 = np.random.randn(2, hidden_units) * scale
    b1 = np.random.randn(hidden_units) * scale
    W2 = np.random.randn(hidden_units, 1) * scale
    b2 = np.random.randn(1) * scale
    return W1, b1, W2, b2


def train_xor(X, y, hidden_units=2, learning_rate=0.5, n_epochs=10000, seed=42, noise_std=0.0):
    X_train = X.copy()
    if noise_std > 0:
        np.random.seed(seed)
        X_train = X_train + np.random.randn(*X_train.shape) * noise_std

    W1, b1, W2, b2 = initialize_parameters(hidden_units=hidden_units, seed=seed, scale=0.5)
    losses = []

    for epoch in range(n_epochs):
        z1 = X_train @ W1 + b1
        a1 = sigmoid(z1)

        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        y_pred = a2.flatten()
        loss = compute_loss_bce(y, y_pred)
        losses.append(loss)

        error2 = y_pred - y
        dW2 = (a1.T @ error2.reshape(-1, 1)) / len(X_train)
        db2 = np.array([np.mean(error2)])

        error1 = (error2.reshape(-1, 1) @ W2.T) * (a1 * (1 - a1))
        dW1 = (X_train.T @ error1) / len(X_train)
        db1 = np.mean(error1, axis=0)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 2000 == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    return W1, b1, W2, b2, losses, X_train


def plot_decision_boundary(W1, b1, W2, b2, X_plot, y_plot, output_path):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z1g = sigmoid(np.dot(grid, W1) + b1)
    z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, s=100, cmap='RdBu', edgecolors='k')
    plt.title("XOR : frontière de décision du réseau 2-2-1")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_experiment(name, hidden_units=2, learning_rate=0.5, n_epochs=10000, seed=42, noise_std=0.0, output_path="phase3_xor_boundary.png"):
    print(name)
    if hidden_units == 1:
        # Avec une seule unité cachée, le réseau ne dispose que d'une seule projection intermédiaire.
        # Cela limite fortement les formes de frontière possibles et ne suffit généralement pas à séparer XOR.
        print("Note : architecture 2-1-1, capacité de représentation insuffisante pour XOR complet.")

    W1, b1, W2, b2, losses, X_train = train_xor(
        X_xor,
        y_xor,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        seed=seed,
        noise_std=noise_std,
    )

    y_pred = sigmoid((sigmoid(X_train @ W1 + b1) @ W2 + b2)).flatten()
    final_loss = losses[-1]
    final_acc = np.mean((y_pred > 0.5) == y_xor)
    plot_decision_boundary(W1, b1, W2, b2, X_train, y_xor, output_path)

    print(f"\nLoss finale : {final_loss:.4f}")
    print(f"Accuracy finale : {final_acc:.2%}")
    print(f"Frontière sauvegardée : {output_path}")
    print()

    return final_loss, final_acc


def main():
    run_experiment("Scénario normal", hidden_units=2, learning_rate=0.5, n_epochs=10000, seed=42, output_path="phase3_xor_boundary.png")
    run_experiment("Cas limite - couche cachée réduite à 1 neurone", hidden_units=1, learning_rate=0.5, n_epochs=10000, seed=42, output_path="phase3_xor_boundary_hidden1.png")
    run_experiment("Scénario adversarial - bruit de 5%", hidden_units=2, learning_rate=0.5, n_epochs=10000, seed=42, noise_std=0.05, output_path="phase3_xor_boundary_noisy.png")


if __name__ == "__main__":
    main()