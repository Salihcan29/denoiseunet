from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
from src.denoisenn.denoise_dataset import DenoiseDataset


def train_model(model, noisy_dir, clean_dir, epochs=10, batch_size=32, lr=0.001):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = DenoiseDataset(noisy_dir=noisy_dir, clean_dir=clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for noisy_images, clean_images in dataloader:
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    print("Training completed.")