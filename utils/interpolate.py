import torch
import matplotlib.pyplot as plt

def linear_interpolation(x, x_values, y_values):
    """
    Perform 1D linear interpolation using PyTorch.

    Parameters:
        x (torch.Tensor): The input value(s) for interpolation.
        x_values (torch.Tensor): The x-coordinates of the data points.
        y_values (torch.Tensor): The y-coordinates of the data points.

    Returns:
        torch.Tensor: Interpolated values corresponding to the input x.
    """
    
    # Check if input x is scalar or tensor
    is_scalar = x.dim() == 0

    # Calculate the indices for interpolation
    indices = torch.searchsorted(x_values, x, right=True)

    # Clip the indices to be within the valid range
    indices = torch.clamp(indices, 1, len(x_values) - 1)

    # Calculate the slopes and offsets for interpolation
    x_left = x_values[indices - 1]
    x_right = x_values[indices]
    y_left = y_values[indices - 1]
    y_right = y_values[indices]
    slope = (y_right - y_left) / (x_right - x_left)
    offset = y_left - slope * x_left

    # Perform linear interpolation
    interpolated_values = slope * x + offset

    # If input x was a scalar, return the scalar result
    if is_scalar:
        interpolated_values = interpolated_values.item()

    return interpolated_values

# Example usage:
if __name__ == '__main__':
    x_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_values = torch.tensor([2.0, 3.0, 5.0, 8.0, 12.0])
    x_input = torch.linspace(1,5,100)
    #x_input = torch.Tensor([1.5])
    
    result = linear_interpolation(x_input, x_values, y_values)
    plt.plot(result)
