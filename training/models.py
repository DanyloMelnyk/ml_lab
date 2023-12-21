import timm
from torch import nn
from torchvision.models.squeezenet import squeezenet1_0


class IdentityWithPrintShape(nn.Module):
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        return x


class CnnLstmModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        n_classes: int,
        lstm_input_size: int,
    ):
        super().__init__()

        self.n_classes = n_classes

        self.feature_extractor = feature_extractor

        self.lstm_input_size = lstm_input_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=128, num_layers=2, batch_first=True
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        B, *_ = x.shape

        x = self.feature_extractor.forward_features(x)

        x = x.view(B, -1, self.lstm_input_size)

        x, *_ = self.lstm(x)
        x = x[:, -1, :]

        x = self.head(x)

        return x


class SqueezeNetWrapper(nn.Module):
    def __init__(self, squeeze_net: nn.Module):
        super().__init__()
        self.squeeze_net = squeeze_net

    def forward(self, x):
        return self.squeeze_net(x)

    def forward_features(self, x):
        return self.squeeze_net.features(x)


def create_densenet_model(with_lstm):
    model = timm.create_model("densenet121", pretrained=True, num_classes=3)

    for name, param in model.named_parameters():
        if not name.startswith(
            (
                "features.denseblock4.denselayer14",
                "features.denseblock4.denselayer15",
                "features.denseblock4.denselayer16",
                "features.norm5",
                "classifier",
            )
        ):
            param.requires_grad = False

    if with_lstm:
        return CnnLstmModel(model, 3, 7 * 7)
    else:
        return model


def create_squeezenet_model(with_lstm):
    model = squeezenet1_0(pretrained=True)

    final_conv = nn.Conv2d(512, 3, kernel_size=1)
    nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
    model.classifier[1] = final_conv

    for name, param in model.named_parameters():
        if not name.startswith(
            (
                "features.12",
                "classifier",
            )
        ):
            param.requires_grad = False

    if with_lstm:
        return CnnLstmModel(SqueezeNetWrapper(model), 3, 13 * 13)
    else:
        return model
