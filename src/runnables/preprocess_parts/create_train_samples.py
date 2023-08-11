import cv2
import os
import numpy as np

"""
orijinal ve gürültülü png dosyalarından samplelar üretir ve kayıt eder.
pipeline üzerinden çalıştırıldığı için aşağıdaki path bilgileri güncellenebilir
"""


def save_random_crops(original_image, noisy_image, x_train_dir, y_train_dir, idx, k=64):
    for i in range(200):
        # Rastgele bir başlangıç noktası seç
        x = np.random.randint(original_image.shape[1] - k)
        y = np.random.randint(original_image.shape[0] - k)

        # Her iki resimden de aynı noktaya denk gelen kxk'lük parçaları al
        original_crop = original_image[y:y+k, x:x+k]
        noisy_crop = noisy_image[y:y+k, x:x+k]

        # Parçaları kaydet
        cv2.imwrite(os.path.join(y_train_dir, str(idx) + '.png'), original_crop)
        cv2.imwrite(os.path.join(x_train_dir, str(idx) + '.png'), noisy_crop)
        idx += 1

    return idx
