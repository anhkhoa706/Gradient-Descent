import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Training parameters
LEARNING_RATE = 1e-3  # Adjusted for better convergence
EPOCHS = 10000        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
DATA_PATH = config["data_path"]

# Define the model
class HousePricePredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)  # Input dimensions dynamically set

    def forward(self, x):
        return self.linear(x)

def normalize_data(tensor: torch.Tensor) -> torch.Tensor:   
    """Normalizes the tensor using mean and standard deviation."""
    return (tensor - tensor.mean()) / tensor.std()

def load_data(file_path):
    """
    Load CSV data and convert selected columns to torch tensors.
    """
    df = pd.read_csv(file_path)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Selecting input features (excluding 'the unit price (NTD / square meter)')
    feature_columns = ['main building area', 'auxiliary building area', 'balcony area', 
                       'property age', 'distance to city center', 'area (square meters)']
    
    target_column = 'total price NTD'  # Ensure this matches exactly

    # Extract features and target variable
    features = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)

    # Convert to tensors
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32).view(-1, 1)  # Reshape for single-output model

    # Normalize features
    features = normalize_data(features)

    return features.to(DEVICE), target.to(DEVICE)

def train_model(model, features, target, epochs, learning_rate):
    """
    Train the linear regression model using Mean Squared Error loss and SGD optimizer.
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    return losses

def evaluate_model(model, features, target):
    """
    Evaluate the model and visualize predictions vs actual values.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(features).cpu().numpy()
        actual = target.cpu().numpy()

    # Scatter plot (Actual vs. Predicted)
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, alpha=0.6, color='blue', label="Predictions")
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label="Perfect Fit")
    plt.xlabel("Actual Total Price")
    plt.ylabel("Predicted Total Price")
    plt.title("Actual vs Predicted Total Price")
    plt.legend()
    plt.show()

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

def plot_loss(losses):
    """
    Plot the training loss over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Training")
    plt.show()

def main():
    # Load the data from the CSV file
    data_file = f"{DATA_PATH}\processed\selected_columns.csv"
    features, target = load_data(data_file)

    # Initialize the model with the correct number of features
    input_dim = features.shape[1]
    model = HousePricePredictor(input_dim).to(DEVICE)

    # Train the model
    losses = train_model(model, features, target, EPOCHS, LEARNING_RATE)

    # Extract trained weights
    weights = model.linear.weight[0].tolist()
    bias = model.linear.bias.item()

    print(f'Final weights: {weights}')
    print(f'Final bias: {bias}')
    
    # Plot the training loss curve
    plot_loss(losses)

    # Evaluate the model with visualization
    evaluate_model(model, features, target)

if __name__ == '__main__':
    main()
