import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train(X, y, learning_rate=0.1, n_epochs=50, seed=42, output_path="phase2_loss_curve.png"):
    np.random.seed(seed)
    w = np.random.randn(2) * 0.01
    b = 0.0
    losses = []

    for epoch in range(n_epochs):
        z = X @ w + b
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        error = y_pred - y
        dw = (X.T @ error) / len(X)
        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss BCE")
    plt.title("Convergence du neurone unique")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\nCourbe sauvegardée : {output_path}")
    print(f"Loss finale : {losses[-1]:.4f}")

    return {
        "w": w,
        "b": b,
        "losses": losses,
        "y_pred": sigmoid(X @ w + b),
    }


def run_scenario(label, learning_rate, output_path):
    print(label)
    result = train(X, y, learning_rate=learning_rate, output_path=output_path)
    print()
    return result


def main():
    run_scenario("Scénario normal", learning_rate=0.4, output_path="phase2_loss_curve.png")
    run_scenario("Cas limite - learning_rate nul", learning_rate=0.0, output_path="phase2_loss_curve_lr0.png")
    run_scenario("Scénario adversarial - learning_rate trop grand", learning_rate=10.0, output_path="phase2_loss_curve_lr10.png")


if __name__ == "__main__":
    main()