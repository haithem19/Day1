import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_spiral(n_points=200, noise=0.1, seed=42):
    """Génère deux spirales entrelacées : classe 0 et classe 1."""
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def init_layer(n_in, n_out, seed_offset):
    std = np.sqrt(2 / n_in)
    rng = np.random.default_rng(42 + seed_offset)
    return rng.standard_normal((n_in, n_out)) * std, np.zeros(n_out)


def init_network(hidden_sizes):
    W1, b1 = init_layer(2, hidden_sizes[0], 1)
    W2, b2 = init_layer(hidden_sizes[0], hidden_sizes[1], 2)
    W3, b3 = init_layer(hidden_sizes[1], 1, 3)
    return W1, b1, W2, b2, W3, b3


def train_spiral(X, y, hidden_sizes=(64, 64), lr=0.01, n_epochs=2000, noise_label=None, output_path="phase4_spirale.png"):
    W1, b1, W2, b2, W3, b3 = init_network(hidden_sizes)
    losses = []

    for epoch in range(n_epochs):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        y_pred = sigmoid(z3).flatten()

        loss = bce_loss(y, y_pred)
        losses.append(loss)

        err3 = y_pred - y
        dW3 = (a2.T @ err3.reshape(-1, 1)) / len(X)
        db3 = np.array([np.mean(err3)])

        err2 = (err3.reshape(-1, 1) @ W3.T) * relu_grad(z2)
        dW2 = (a1.T @ err2) / len(X)
        db2 = np.mean(err2, axis=0)

        err1 = (err2 @ W2.T) * relu_grad(z1)
        dW1 = (X.T @ err1) / len(X)
        db1 = np.mean(err1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        if epoch % 500 == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    h = 0.02
    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
        np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    a1g = relu(np.dot(grid, W1) + b1)
    a2g = relu(np.dot(a1g, W2) + b2)
    zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='none')
    title = "Frontière de décision (2-64-64-1)" if hidden_sizes == (64, 64) else "Frontière de décision (2-2-1)"
    if noise_label:
        title = f"{title} - {noise_label}"
    axes[0].set_title(title)
    axes[1].plot(losses)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss BCE")
    axes[1].set_title("Courbe de loss spirale")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    final_acc = np.mean((y_pred > 0.5) == y)
    print(f"\nLoss finale : {losses[-1]:.4f}")
    print(f"Accuracy finale : {final_acc:.2%}")
    print(f"Courbe sauvegardée : {output_path}")
    print()
    return losses, final_acc


def main():
    X, y = generate_spiral(n_points=400, noise=0.15)
    print("Scénario normal")
    _, acc_clean = train_spiral(X, y, hidden_sizes=(64, 64), lr=0.1, n_epochs=10000, output_path="phase4_spirale.png")

    print("Cas limite - architecture 2-2-1")
    _, acc_underfit = train_spiral(X, y, hidden_sizes=(2, 2), lr=0.1, n_epochs=10000, output_path="phase4_spirale_underfit.png")
    print(f"Comparaison accuracy | 2-64-64-1: {acc_clean:.2%} | 2-2-1: {acc_underfit:.2%}")

    print("Scénario adversarial - bruit 0.5")
    X_noisy, y_noisy = generate_spiral(n_points=400, noise=0.5)
    train_spiral(X_noisy, y_noisy, hidden_sizes=(64, 64), lr=0.1, n_epochs=10000, noise_label="bruit 0.5", output_path="phase4_spirale_noise.png")


if __name__ == "__main__":
    main()