import torch
from src.denoisenn.engine import DenoiseUNet
from src.denoisenn.trainer import train_model
import time

import src.runnables.run_preprocess_pipeline

src.runnables.run_preprocess_pipeline.run(crops_per_pdf=250, image_size=128)

noisy_dir = './data/dataset/x_train'
clean_dir = './data/dataset/y_train'

# model = DenoiseUNet()
model = torch.load('model_backup/DenoiseUnet_128x128_50ep_8batch.pt')

train_model(model=model, noisy_dir=noisy_dir, clean_dir=clean_dir, epochs=10, batch_size=16, lr=0.00001)

torch.save(model, 'DenoiseUnet.pt')
