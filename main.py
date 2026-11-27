"""
Backpropagation from scratch: training and visualisation.
- Generates 2-D “two moons” data
- Checks the gradient numerically
- Trains a small multi layer perceptron (MLP) with hand-written backprop
- Saves the evolution of the decision boundary across epochs
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mlp import MLP, binary_cross_entropy


def make_moons(n=200, noise=0.12, seed=42):
    """Two half-moons – classic non-linear classification task."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out

    # outer half-moon
    theta = np.linspace(0, np.pi, n_out)
    x1 = np.cos(theta) + rng.normal(0, noise, n_out)
    y1 = np.sin(theta) + rng.normal(0, noise, n_out)

    # inner half-moon (shifted)
    theta = np.linspace(0, np.pi, n_in)
    x2 = 1 - np.cos(theta) + rng.normal(0, noise, n_in)
    y2 = 0.5 - np.sin(theta) + rng.normal(0, noise, n_in)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).reshape(-1, 1)
    # shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


def numerical_gradients(model, X, y, h=1e-5):
    """Central difference for every weight – gradient checking."""
    params = model.parameters()
    grads_num = []
    for p in params:
        grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + h
            loss_plus = binary_cross_entropy(y, model.forward(X))
            p[idx] = old - h
            loss_minus = binary_cross_entropy(y, model.forward(X))
            grad[idx] = (loss_plus - loss_minus) / (2 * h)
            p[idx] = old
            it.iternext()
        grads_num.append(grad)
    return grads_num


def relative_error(a, b, eps=1e-8):
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + eps))


def plot_decision_boundary(model, X, y, ax, title=""):
    """Draw decision boundary and data points."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid).reshape(xx.shape)

    cmap_pts = ListedColormap(["#c0392b", "#2980b9"])
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=cmap_pts, edgecolors="k", s=28, linewidths=0.4)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def main():
    X, y = make_moons(n=300, noise=0.15)
    model = MLP(hidden=20, seed=1)

    # --- gradient check ---
    print("Gradient check (relative error)...")
    model.forward(X)
    grads_analytic = model.backward(y)
    grads_numeric = numerical_gradients(model, X, y)
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    for name, ga, gn in zip(names, grads_analytic, grads_numeric):
        err = relative_error(ga, gn)
        print(f"  {name:4s}  max rel. error = {err:.2e}")
        assert err < 1e-4, f"Gradient for {name} does not match!"

    # --- training ---
    lr = 0.8
    epochs = 400
    snapshot_epochs = [0, 20, 50, 100, 200, 400]
    losses = []
    snapshots = {}

    # initial state before training
    snapshots[0] = (model.W1.copy(), model.b1.copy(), model.W2.copy(),
                    model.b2.copy(), model.W3.copy(), model.b3.copy())

    for epoch in range(1, epochs + 1):
        proba = model.forward(X)
        loss = binary_cross_entropy(y, proba)
        losses.append(loss)
        grads = model.backward(y)
        for p, g in zip(model.parameters(), grads):
            p -= lr * g

        if epoch in snapshot_epochs:
            snapshots[epoch] = (model.W1.copy(), model.b1.copy(), model.W2.copy(),
                                model.b2.copy(), model.W3.copy(), model.b3.copy())

        if epoch % 50 == 0 or epoch == 1:
            acc = np.mean((proba >= 0.5) == y)
            print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")

    # --- decision-boundary evolution ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.ravel()
    for ax, ep in zip(axes, snapshot_epochs):
        # load snapshot
        W1, b1, W2, b2, W3, b3 = snapshots[ep]
        model.W1, model.b1 = W1, b1
        model.W2, model.b2 = W2, b2
        model.W3, model.b3 = W3, b3
        plot_decision_boundary(model, X, y, ax, title=f"epoch {ep}")
    fig.suptitle("Evolution of the decision boundary during training", fontsize=14, y=0.97) #1.01
    fig.tight_layout()
    fig.savefig("decision_boundaries.png", dpi=500, bbox_inches="tight", pad_inches=0.01)
    print("Saved decision_boundaries.png")

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color="#2c3e50", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy")
    ax.set_title("Loss during training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=150)
    print("Saved loss_curve.png")

    # final accuracy
    final_acc = np.mean(model.predict(X) == y)
    print(f"Final accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    main()
