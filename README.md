## 🏗️ Mathematical Integration & Data Flow Architecture

```mermaid
graph TD
    A["Raw Data: tmax, tmin, rain"] -->|"StandardScaler: z = x - μ / σ"| B["Normalized Input Batch X"]
    B -->|"Forward Pass: Z = XW + b"| C["Linear Transformation"]
    C -->|"ReLU: A = max(0, Z)"| D["Hidden Layers Activations"]
    D -->|"Final Layer Output y_hat"| E["MSE Loss & Gradient: ŷ - y"]
    E -->|"Backward Pass: ReLU Derivative"| F["Error Propagation & Chain Rule"]
    F -->|"Weight & Bias Gradients"| G["Parameter Updates: W = W - α * w_grad"]