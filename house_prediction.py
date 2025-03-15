import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Training parameters
LEARNING_RATE = 1e-7  # 0.0000001
EPOCHS = 5000         # Number of epochs

# Define the model
class HousePricePredictor(torch.nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, main_area, balcony_area):
        return self.w1 * main_area + self.w2 * balcony_area

def load_data(file_path):
    """
    Load CSV data and convert selected columns to torch tensors.
    """
    df = pd.read_csv(file_path)
    main_area = torch.tensor(df['main building area'].values, dtype=torch.float32)
    balcony_area = torch.tensor(df['balcony area'].values, dtype=torch.float32)
    unit_price = torch.tensor(df['the unit price (NTD / square meter)'].values, dtype=torch.float32)
    return main_area, balcony_area, unit_price

def train_model(model, main_area, balcony_area, unit_price, epochs, learning_rate):
    """
    Train the linear regression model using Mean Squared Error loss and SGD optimizer.
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(main_area, balcony_area)
        loss = loss_fn(predictions, unit_price)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    return losses

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

def plot_regression_plane(model, main_area, balcony_area, unit_price):
    """
    Plot the original data and the regression plane in a 3D scatter plot.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original data points
    ax.scatter(main_area.numpy(), balcony_area.numpy(), unit_price.numpy(), 
               color='blue', label='Original data')
    
    # Create mesh grid for regression plane
    main_min, main_max = main_area.min().item(), main_area.max().item()
    balcony_min, balcony_max = balcony_area.min().item(), balcony_area.max().item()
    main_range = np.linspace(main_min, main_max, 10)
    balcony_range = np.linspace(balcony_min, balcony_max, 10)
    main_grid, balcony_grid = np.meshgrid(main_range, balcony_range)
    
    # Calculate predicted prices using the trained weights
    price_grid = model.w1.item() * main_grid + model.w2.item() * balcony_grid
    
    # Plot the regression plane
    ax.plot_surface(main_grid, balcony_grid, price_grid, color='red', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Main building area')
    ax.set_ylabel('Balcony area')
    ax.set_zlabel('Unit Price')
    ax.set_title('Multiple Linear Regression: House Price Prediction')
    ax.legend()
    plt.show()

def main():
    # Load the data from the CSV file
    data_file = "csv/selected_columns_remove_zero.csv"
    main_area, balcony_area, unit_price = load_data(data_file)
    
    # Initialize the model
    model = HousePricePredictor()
    
    # Train the model
    losses = train_model(model, main_area, balcony_area, unit_price, EPOCHS, LEARNING_RATE)
    print(f'Final weights: w1 = {model.w1.item()}, w2 = {model.w2.item()}')
    
    # Plot the training loss curve
    plot_loss(losses)
    
    # Plot the 3D regression plane along with original data
    plot_regression_plane(model, main_area, balcony_area, unit_price)

if __name__ == '__main__':
    main()
