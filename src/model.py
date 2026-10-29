import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

#Load  weather data
np.random.seed(42)
n_rows = 200
data = pd.DataFrame(
    {
        "tmax": np.random.uniform(20, 40, n_rows),
        "tmin": np.random.uniform(10, 20, n_rows),
        "rain": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        "tmax_tomorrow": np.random.uniform(20, 40, n_rows),
    }
)

PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

split_data = np.split(data, [int(0.7 * len(data)), int(0.85 * len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [
    [d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data
]

# Convert NumPy arrays to PyTorch Tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
valid_x = torch.tensor(valid_x, dtype=torch.float32)
valid_y = torch.tensor(valid_y, dtype=torch.float32)


#  Define  PyTorch Neural Network Class
class WeatherNN(nn.Module):

    def __init__(self):
        super(WeatherNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


# Initialize model, loss function, and optimizer
model = WeatherNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#  Training Loop
epochs = 10
batch_size = 8

print("Starting Authentic PyTorch Training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batches_count = 0

    for i in range(0, train_x.shape[0], batch_size):
        x_batch = train_x[i : i + batch_size]
        y_batch = train_y[i : i + batch_size]

        # Forward pass: compute predictions
        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        # Backward pass & optimization (PyTorch handles gradients automatically)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches_count += 1

    # Validation pass
    model.eval()
    with torch.no_grad():
        val_preds = model(valid_x)
        val_loss = criterion(val_preds, valid_y)

    train_loss = epoch_loss / batches_count
    print(
        f"Epoch {epoch} | Train Loss (MSE): {train_loss:.4f} | Valid Loss (MSE): {val_loss.item():.4f}"
    )