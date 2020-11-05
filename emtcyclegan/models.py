from torch import nn


class CompensationModel(torch.nn.Module):
    def __init__(self, input_features, output_features, units, activation=nn.LeakyReLU(0.01)):
        super(CompensationModel, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layers = []
        last_size = len(input_features)
        for i, u in enumerate(units):
            layer = nn.Sequential(nn.Linear(last_size, u), activation)
            last_size = u
            self.layers.append(layer)
        final_layer = nn.Sequential(nn.Linear(last_size, len(output_features)))
        self.layers.append(final_layer)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, input_features, units, activation=nn.LeakyReLU(0.2)):
        super(DiscriminatorModel, self).__init__()
        self.input_features = input_features
        self.layers = []
        last_size = len(input_features)
        for i, u in enumerate(units):
            layer = nn.Sequential(nn.Linear(last_size, u), activation)
            last_size = u
            self.layers.append(layer)
        final_layer = nn.Sequential(nn.Linear(last_size, nn.Sigmoid())

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
