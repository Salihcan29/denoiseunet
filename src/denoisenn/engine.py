from torch import nn


class DenoiseNN(nn.Module):
    def __init__(self, window_size=64):
        super(DenoiseNN, self).__init__()
        self.window_size = window_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, window_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(window_size, window_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(window_size, window_size,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Conv2d(window_size, window_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(window_size, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

