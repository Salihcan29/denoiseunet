import cv2
import torch
import numpy as np
from PIL import Image

from src.denoisenn.engine import DenoiseNN
from src.runnables import run_preprocess_pipeline

# run_preprocess_pipeline.run(image_size=128)


def test_model(model, image_path, k=64):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    result_image = np.zeros_like(image)

    model.eval()

    for y in range(0, image.shape[0], k):
        for x in range(0, image.shape[1], k):
            crop = image[y:y+k, x:x+k]

            input_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_crop = (output_tensor.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8)
            result_image[y:y+k, x:x+k] = output_crop

    return result_image


def denoise_image(model_path, image_path, window_size):
    # model = DenoiseNN()
    model = torch.load(model_path)

    # model.load_state_dict(torch.load(model_path))
    model.eval()

    result_image_pil = test_model(model, image_path, window_size)

    return result_image_pil


model_path = 'DenoiseNN.pt'

original_path = r'C:\Users\can.turan\PycharmProjects\artiwisedenoisenn\data\dataset\y_train\0.png'
image_path = r'C:\Users\can.turan\PycharmProjects\artiwisedenoisenn\data\dataset\x_train\0.png'

window_size = 128

input_image = cv2.imread(image_path)
result_image = denoise_image(model_path, image_path, window_size)
original_image = cv2.imread(original_path)

cv2.imshow("input", cv2.resize(input_image, (window_size, window_size)))
cv2.imshow("image", cv2.resize(result_image, (window_size, window_size)))
cv2.imshow("orig", cv2.resize(original_image, (window_size, window_size)))

cv2.waitKey(0)
