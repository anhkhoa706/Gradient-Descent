# Gradient-Descent
# House Price Prediction

This project implements a machine learning model using PyTorch to predict house prices based on various features. The model is a simple linear regression trained with gradient descent.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [License](#license)

## Project Overview
The goal of this project is to predict house prices based on key property features such as building area, property age, and distance to the city center. The model is trained using a linear regression approach.

## Dataset
The dataset is stored in a CSV file (`selected_columns.csv`) and contains the following features:
- **main building area**
- **auxiliary building area**
- **balcony area**
- **property age**
- **distance to city center**
- **area (square meters)**
- **total price NTD** (target variable)

## Installation
Ensure you have Python and the required libraries installed:

```sh
pip install torch pandas numpy matplotlib
```

## Usage
To train and evaluate the model, run:

```sh
python house_price_prediction.py
```

The script will:
1. Load and preprocess the dataset.
2. Train a linear regression model using Mean Squared Error loss and Stochastic Gradient Descent (SGD).
3. Display training loss and model evaluation metrics.
4. Generate visualization for actual vs. predicted house prices.

## Model
The model is a simple **linear regression** implemented in PyTorch:

```python
class HousePricePredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
```

The training loop minimizes Mean Squared Error (MSE) loss using an SGD optimizer.

## Results
- The training loss curve is displayed.
- The model's performance is evaluated using **Mean Absolute Percentage Error (MAPE)**.
- A scatter plot shows the comparison between actual and predicted prices.

## License
This project is licensed under the MIT License.