import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.denoisenn.engine import DenoiseUNet
from src.runnables import run_preprocess_pipeline

# run_preprocess_pipeline.run(image_size=128)

def test_model(model, image, k=64):
    result_image = np.zeros_like(image)

    model.eval()

    for y in range(0, image.shape[0], k):
        for x in range(0, image.shape[1], k):
            crop = image[y:min(y+k, image.shape[0]), x:min(x+k, image.shape[1])]

            # Eğer parça k x k değilse, sıfırlarla doldur
            if crop.shape != (k, k):
                padded_crop = np.zeros((k, k), dtype=crop.dtype)
                padded_crop[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded_crop

            input_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_tensor[output_tensor > 1] = 1.
            output_crop = (output_tensor.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8)
            result_image[y:min(y+k, image.shape[0]), x:min(x+k, image.shape[1])] = output_crop[:crop.shape[0], :crop.shape[1]]

    return result_image


def denoise_image(model_path, image_path, window_size):
    # model = DenoiseNN()
    model = torch.load(model_path)

    # model.load_state_dict(torch.load(model_path))
    model.eval()

    result_image_pil = test_model(model, image_path, window_size)

    return result_image_pil


model_path = 'model_backup/DenoiseUnet_128x128_50ep_8batch.pt'
model_path = 'DenoiseUnet.pt'


_id = 555

original_path = f'./data/dataset/y_train/{_id}.png'
image_path = f'./data/dataset/x_train/{_id}.png'

window_size = 128

input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # [:,::-1].copy()
result_image = denoise_image(model_path, input_image, window_size)
original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

display_size = 128

# cv2.imshow("noised", cv2.resize(input_image, (display_size, display_size)))
# cv2.imshow("prediction", cv2.resize(result_image, (display_size, display_size)))
# cv2.imshow("ground_truth", cv2.resize(original_image, (display_size, display_size)))
# cv2.waitKey(0)

plt.subplot(1,3,1)
plt.title('Noised')
plt.imshow(cv2.resize(input_image, (display_size, display_size)), cmap='gray')

plt.subplot(1,3,2)
plt.title('Denoised')
plt.imshow(cv2.resize(result_image, (display_size, display_size)), cmap='gray')

plt.subplot(1,3,3)
plt.title('Original')
plt.imshow(cv2.resize(original_image, (display_size, display_size)), cmap='gray')
plt.show()


