import numpy as np

"""
Bu çalıştırılabilir kod dosyası kaynak dizindeki tüm pdfleri png formatına çevirir

source: ./data/preprocess/original_pngs/
target: ./data/preprocess/noisy_pngs/
"""


class NoiseGenerator:
    def __init__(self):
        pass

    def addPixels(self, image, blackProb, whiteProb):
        h, w = image.shape

        random_numbers = np.random.rand(h, w)

        black_pixels = random_numbers < blackProb
        white_pixels = (random_numbers >= blackProb) & (random_numbers < blackProb+whiteProb)

        image[black_pixels] = 0
        image[white_pixels] = 255
