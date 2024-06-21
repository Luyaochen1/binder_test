# Install required libraries
!pip install cloud-tpu-client torch-xla

# Import necessary libraries
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Define a simple operation to test TPU
def test_tpu():
    # Acquires the default Cloud TPU core and moves the computation to it
    device = xm.xla_device()

    # Create tensors
    tensor1 = torch.randn(3, 3, device=device)
    tensor2 = torch.randn(3, 3, device=device)

    # Perform a matrix multiplication
    result = torch.mm(tensor1, tensor2)

    # Print the result
    print('Result of matrix multiplication on TPU:')
    print(result)

# Run the test function
test_tpu()
