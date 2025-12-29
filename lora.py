import torch
import torch.nn as nn

"""
Neural Network with LoRA Architecture:

Input (10)
    │
    ▼
┌──────────────────┐      ┌──────────────────────────────┐
│  model.fc1 (10)  │ ───▶ │ loralayer1 (A:10x4, B:4x20)  │
└──────────────────┘      └──────────────────────────────┘
    │                             │
    └───────────────┬─────────────┘
                    ▼
                  ReLU
                    │
                    ▼
┌──────────────────┐      ┌──────────────────────────────┐
│  model.fc2 (20)  │ ───▶ │ loralayer2 (A:20x4, B:4x20)  │
└──────────────────┘      └──────────────────────────────┘
    │                             │
    └───────────────┬─────────────┘
                    ▼
                  ReLU
                    │
                    ▼
┌──────────────────┐      ┌──────────────────────────────┐
│  model.fc3 (20)  │ ───▶ │ loralayer3 (A:20x4, B:4x20)  │
└──────────────────┘      └──────────────────────────────┘
    │                             │
    └───────────────┬─────────────┘
                    ▼
                  ReLU
                    │
                    ▼
┌──────────────────┐
│  model.fc4 (20)  │ (Frozen, No LoRA)
└──────────────────┘
    │
    ▼
Output (10)

Feature	lora.py
Logic	$W + (A \times B)$
Layers	Simple nn.Linear
Parameters	~1,000
Training	Trains $A$ and $B$
"""

class LoRAWeights(nn.Module):
    """
    A class that implements the LoRA weights.
    """
    def __init__(self, d, k, r, alpha):
        """
        Initializes the LoRA weights.
        :param d: The number of rows in matrix W.
        :param k: The number of columns in matrix W.
        :param r: The rank hyperparameter.
        :param alpha: A scaling parameter that controls the strength of the adaptation.
        """
        super(LoRAWeights, self).__init__()
        # Matrix A is initialized from a Gaussian distribution, it has dimensions d x r
        self.A = nn.Parameter(torch.randn(d, r))
        # Matrix B is initialized as a zero matrix, it has dimensions r x k
        self.B = nn.Parameter(torch.zeros(r, k))
        # Alpha is a scaling parameter that controls the strength of the adaptation, it is a scalar
        self.alpha = alpha

    def forward(self, x):
        # The input x is multiplied by A and B, then scaled by alpha, it has dimensions d x k
        x = self.alpha * (x @ self.A @ self.B)
        return x

class MyNeuralNetwork(nn.Module):
    """
    A simple neural network with 4 linear layers and ReLU activation functions.
    """
    def __init__(self):
        """
        Initializes the neural network.
        """
        super(MyNeuralNetwork, self).__init__()
        # there are 4 layers in the network
        # fc1: 10 -> 20
        # fc2: 20 -> 20
        # fc3: 20 -> 20
        # fc4: 20 -> 10
        # the last layer is a linear layer that outputs the final result
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20) # this is the last layer
        self.fc4 = nn.Linear(20, 10) # this is the output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MyNeuralNetworkWithLoRA(nn.Module):
    def __init__(self, model, r, alpha):
        """
        Initializes the neural network with LoRA.
        """
        super(MyNeuralNetworkWithLoRA, self).__init__()
        self.model = model

        # Create LoRA layers for fc1, fc2, fc3 based on their dimensions
        # fc1: 10 -> 20, so d=10, k=20
        self.loralayer1 = LoRAWeights(10, 20, r, alpha)
        # fc2: 20 -> 20, so d=20, k=20
        self.loralayer2 = LoRAWeights(20, 20, r, alpha)
        # fc3: 20 -> 20, so d=20, k=20
        self.loralayer3 = LoRAWeights(20, 20, r, alpha)

    def forward(self, x):
        # Pass input through original layer and add LoRA output, then apply ReLU
        x = torch.relu(self.model.fc1(x) + self.loralayer1(x))
        x = torch.relu(self.model.fc2(x) + self.loralayer2(x))
        x = torch.relu(self.model.fc3(x) + self.loralayer3(x))
        # fc4 is kept as is
        x = self.model.fc4(x)
        return x

if __name__ == "__main__":
    # 1. Instantiate the original model
    model = MyNeuralNetwork()

    # 2. Freeze the original model parameters
    for param in model.parameters():
        param.requires_grad = False

    # 3. Instantiate the LoRA model
    r = 4
    alpha = 1.0
    lora_model = MyNeuralNetworkWithLoRA(model, r, alpha)

    # 4. Verify which parameters are trainable
    print("Trainable parameters:")
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # 5. Test forward pass
    test_input = torch.randn(1, 10)
    output = lora_model(test_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output: {output}")
