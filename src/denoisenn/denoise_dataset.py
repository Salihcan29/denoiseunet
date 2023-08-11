from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image


class DenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.image_filenames = os.listdir(noisy_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        noisy_image = Image.open(os.path.join(self.noisy_dir, self.image_filenames[idx]))
        clean_image = Image.open(os.path.join(self.clean_dir, self.image_filenames[idx]))

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

