import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate Synthetic Dataset (Standalone)
np.random.seed(42)
n_samples = 300

data = pd.DataFrame({
    "tmax": np.random.uniform(15, 35, n_samples),
    "tmin": np.random.uniform(5, 20, n_samples),
    "rain": np.random.choice([0, 1, 5, 12], size=n_samples, p=[0.7, 0.15, 0.1, 0.05]),
    "tmax_tomorrow": np.random.uniform(15, 35, n_samples)
})

# Data Splitting
PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"  # output

scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

split_data = np.split(data, [int(0.7 * len(data)), int(0.85 * len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [
    [d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data
]

#  Neural Network Architecture 
def init_layers(inputs):
    layers = []
    for i in range(1, len(inputs)):
        layers.append(
            [
                np.random.rand(inputs[i - 1], inputs[i]) / 5 - 0.1,
                np.ones((1, inputs[i])),
            ]
        )
    return layers 

layer_conf = [3, 10, 10, 1]
layers = init_layers(layer_conf)

# Forward pass
def forward(batch, layers):
    hiddens = [batch.copy()]
    for i in range(len(layers)):
        batch = np.matmul(batch, layers[i][0]) + layers[i][1]
        if i < len(layers) - 1:
            batch = np.maximum(batch, 0)  # ReLU activation
        hiddens.append(batch.copy())
    return hiddens 

# Loss Functions
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def mse_grad(actual, predicted):
    return predicted - actual

# Backward Pass
def backward(layers, hiddens, grad, lr):
    batch_size = hiddens[0].shape[0]
    for i in range(len(layers) - 1, -1, -1):
        if i != len(layers) - 1:
            grad = np.multiply(grad, np.heaviside(hiddens[i + 1], 0))
        
        # Normalized weight gradients by batch size to match bias scaling
        w_grad = (hiddens[i].T @ grad) / batch_size  
        b_grad = np.mean(grad, axis=0)
        
        layers[i][0] -= lr * w_grad
        layers[i][1] -= lr * b_grad
        
        if i != 0:
            grad = grad @ layers[i][0].T
    return layers  

#  Training Loop
lr = 0.001  
epochs = 10
batch_size = 8
layers = init_layers(layer_conf)

print("--- Starting Training ---")
for epoch in range(epochs):
    epoch_loss = 0
    batches_count = 0
    for i in range(0, train_x.shape[0], batch_size):
        x_batch = train_x[i : i + batch_size]
        y_batch = train_y[i : i + batch_size]

        hiddens = forward(x_batch, layers)
        pred = hiddens[-1]

        loss_grad = mse_grad(y_batch, pred)
        epoch_loss += mse(y_batch, pred)
        batches_count += 1

        layers = backward(layers, hiddens, loss_grad, lr)

    train_loss = epoch_loss / batches_count
    valid_preds = forward(valid_x, layers)[-1]
    val_loss = mse(valid_y, valid_preds)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("--- Training Complete ---")