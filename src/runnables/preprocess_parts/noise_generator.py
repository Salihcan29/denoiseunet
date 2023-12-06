import numpy as np
import cv2

"""
Bu çalıştırılabilir kod dosyası kaynak dizindeki tüm pdfleri png formatına çevirir

source: ./data/preprocess/original_pngs/
target: ./data/preprocess/noisy_pngs/
"""


class NoiseGenerator:
    def __init__(self):
        pass

    def addPixels(self, image, blackProb, whiteProb):
        image = image.copy()
        h, w = image.shape

        random_numbers = np.random.rand(h, w)

        black_pixels = random_numbers < blackProb
        white_pixels = (random_numbers >= blackProb) & (random_numbers < blackProb+whiteProb)

        image[black_pixels] = 0
        image[white_pixels] = 255
        return image

    def addBlur(self, image, blur_strength):
        # blur_strength 0 ile 1 arasında bir değer olmalıdır.
        # Örneğin 0.1 küçük bir bulanıklık, 1.0 maksimum bulanıklık anlamına gelebilir.
        # Bu örnekte maksimum kernel boyutunu 31 olarak varsayıyorum.
        ksize = int(blur_strength * 30) + 1  # 0.1 için 4, 1.0 için 31 gibi bir değer üretir.
        ksize = max(1, ksize)  # En az 1 olmalıdır.
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Kernel boyutu tek sayı olmalıdır.
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def makeGrayNoiseAreas(self, image, noise_prob):
        h, w = image.shape
        mask = np.random.rand(h, w) < noise_prob
        noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        image[mask] = noise[mask]
        return image

    def addSpeckleNoise(self, image, speckle_intensity):
        noise = np.random.randn(*image.shape) * speckle_intensity
        noisy_image = image + image * noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    def addRandomLines(self, image, line_num, line_thickness, line_color):
        for _ in range(line_num):
            pt1 = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
            pt2 = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
            image = cv2.line(image, pt1, pt2, line_color, line_thickness)
        return image
