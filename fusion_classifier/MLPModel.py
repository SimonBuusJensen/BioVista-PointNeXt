from torch import nn

# Setup MLP model (512 x 256 x 2)
class MLPModel(nn.Module):
    def __init__(self, input_size=1024, output_size=2, option=3, dropout_rate=0.0):
        super(MLPModel, self).__init__()
        
        if option == 1:
            hidden_sizes = [512]  # Simple
        elif option == 2:
            hidden_sizes = [512, 256]  # Deeper
        elif option == 3:
            hidden_sizes = [1024, 512, 256]  # Most expressive
        else:
            raise ValueError("Invalid option. Choose 1, 2, or 3.")

        layers = []
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # Normalization for stability
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # Prevent overfitting
            
            in_features = hidden_size  # Set input size for next layer
        
        layers.append(nn.Linear(in_features, output_size))  # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)