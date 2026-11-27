# Backpropagation

A small multi-layer perceptron (MLP) implemented from scratch with NumPy with forward pass, backpropagation – gradient of the loss and the chain rule – and weight updates. The picture below shows how the decision boundary evolves during training. It is random at epoch 0, starts to separate the two moons after 50 epochs and is smooth by epoch 200. A ReLU network can learn a non-linear boundary that a single linear classifier could never produce.

<div align="center">
  <img src="decision_boundaries.png" alt="decision_boundaries" width="50%">

  <sub>Figure 1: xxx.</sub>
</div>

---

## Relation to Artificial Intelligence

Backpropagation is the algorithm that makes neural networks trainable – from neural networks and transformers to diffusion and reinforcement learning models. When one calls `loss.backward()` in PyTorch, the framework executes the same chain-rule calculations, just for much deeper graphs and with automatic differentiation.

## Mathematics

<ins>Network architecture</ins>

The network maps the input to a probability as

$x \in \mathbb{R}^2 \;\to\; \text{Linear} \to \text{ReLU} \;\to\; \text{Linear} \to \text{ReLU} \;\to\; \text{Linear} \to \sigma \;\to\; \hat y \in (0,1)$.

The parameters of the three linear layers are $W^{(1)}, b^{(1)}$, $W^{(2)}, b^{(2)}$ and $W^{(3)}, b^{(3)}$.

The loss is binary cross-entropy

$L = -\frac1N\sum_{i=1}^N \bigl[\, y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\,\bigr]$.

<ins>Forward pass</ins>

Denote the layers from bottom to top. The forward pass is

$z^{(1)} = XW^{(1)} + b^{(1)},\qquad a^{(1)} = \mathrm{ReLU}(z^{(1)})$

$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)},\qquad a^{(2)} = \mathrm{ReLU}(z^{(2)})$

$z^{(3)} = a^{(2)}W^{(3)} + b^{(3)},\qquad \hat y = a^{(3)} = \sigma(z^{(3)})$.

<ins>Output layer (sigmoid + BCE)</ins>

The combination of sigmoid and binary cross-entropy simplifies nicely as

$\frac{\partial L}{\partial z^{(3)}} = \frac1N\,(\hat y - y)$.

From this we obtain the weight gradients

$\frac{\partial L}{\partial W^{(3)}} = (a^{(2)})^\top \frac{\partial L}{\partial z^{(3)}},\qquad \frac{\partial L}{\partial b^{(3)}} = \sum_i \frac{\partial L}{\partial z^{(3)}_i}$

and the error is sent backwards as

$\frac{\partial L}{\partial a^{(2)}} = \frac{\partial L}{\partial z^{(3)}}\,(W^{(3)})^\top$.

<ins>Hidden layer (ReLU)</ins>

The derivative of ReLU is the indicator function

$\frac{\partial a^{(2)}}{\partial z^{(2)}} = \mathbf{1}_{z^{(2)} > 0}$.

Therefore

$\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \odot \mathbf{1}_{z^{(2)} > 0}$

and, as above,

$\frac{\partial L}{\partial W^{(2)}} = (a^{(1)})^\top \frac{\partial L}{\partial z^{(2)}},\qquad \frac{\partial L}{\partial a^{(1)}} = \frac{\partial L}{\partial z^{(2)}}\,(W^{(2)})^\top$.

The first hidden layer is treated identically (replace $a^{(1)}$ by the input $X$).

<ins>Summary – one backward pass</ins>

1. $\delta^{(3)} = (\hat y - y)/N$
2. $dW^{(3)} = (a^{(2)})^\top\delta^{(3)}$, $\quad db^{(3)} = \sum\delta^{(3)}$
3. $\delta^{(2)} = \bigl(\delta^{(3)}(W^{(3)})^\top\bigr)\odot\mathbf{1}_{z^{(2)}>0}$
4. $dW^{(2)} = (a^{(1)})^\top\delta^{(2)}$, $\quad db^{(2)} = \sum\delta^{(2)}$
5. $\delta^{(1)} = \bigl(\delta^{(2)}(W^{(2)})^\top\bigr)\odot\mathbf{1}_{z^{(1)}>0}$
6. $dW^{(1)} = X^\top\delta^{(1)}$, $\quad db^{(1)} = \sum\delta^{(1)}$

Weights are updated by gradient descent as

$W \leftarrow W - \eta\,dW$.

<ins>Gradient checking</ins>

Code also compares the analytic gradient with the numerical approximation

$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}$.

Relative error is typically below $10^{-6}$. If it grows, there is a bug somewhere.

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Project

```
├── data/               # Benchmark datasets
├── Results/            # Output directory for weights and predictions
├── fm.py               # Main execution script
├── fmloop.py           # Automated experiment runner
├── settings.py         # Hyperparameter configurations
└── statistics.pdf      # Visualisation and analysis
```

## References

Goodfellow, I., Bengio, Y. & Courville, A. (2016). "Back-Propagation and Other Differentiation Algorithms". Deep Learning, chapter 6.5. MIT Press. Available at https://www.deeplearningbook.org/contents/mlp.html.