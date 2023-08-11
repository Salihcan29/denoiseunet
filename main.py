import torch
from src.denoisenn.engine import DenoiseNN
from src.denoisenn.trainer import train_model
import time

noisy_dir = './data/dataset/x_train'
clean_dir = './data/dataset/y_train'

model = DenoiseNN(window_size=128)
train_model(model=model, noisy_dir=noisy_dir, clean_dir=clean_dir, epochs=10, batch_size=32, lr=0.001)

torch.save(model, 'DenoiseNN.pt')

