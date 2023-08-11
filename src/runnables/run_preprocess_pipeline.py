import os
import cv2

"""
source klasöründeki bütün pdfleri okur ve train yapılabilecek şekilde
bütün preprocessing aşamalarını otomatik gerçekleştirir.

Dikkatli kullanılmalıdır original_pdfs klasöründeki dosyalar sabit kalmak üzere
bütün diğer data klasörlerini yeniden generate eder.

source: data/preprocess/original_pdfs
target: multiple outputs
"""

from src.runnables.preprocess_parts.create_train_samples import save_random_crops
from src.runnables.preprocess_parts.noise_generator import NoiseGenerator
from src.runnables.preprocess_parts.pdf_to_png import pdf_to_png


def run(image_size=64):
    # PDF leri png formatına dönüştürme aşaması
    source_dir = './data/preprocess/original_pdfs/'
    target_dir = './data/preprocess/original_pngs/'
    pdf_to_png(source_dir, target_dir)

    source_dir = './data/preprocess/original_pngs/'
    target_dir = './data/preprocess/noisy_pngs/'
    noise_generator = NoiseGenerator()
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            noise_generator.addPixels(image, 0.15, 0.15)

            noisy_image_path = os.path.join(target_dir, filename)
            cv2.imwrite(noisy_image_path, image)

    print("Tüm resim dosyalarına gürültü eklendi.")

    original_dir = './data/preprocess/original_pngs/'
    noisy_dir = './data/preprocess/noisy_pngs/'

    y_train_dir = './data/dataset/y_train/'
    x_train_dir = './data/dataset/x_train/'
    idx = 0
    for filename in os.listdir(original_dir):
        if filename.endswith(".png"):
            original_image = cv2.imread(os.path.join(original_dir, filename), cv2.IMREAD_GRAYSCALE)
            noisy_image = cv2.imread(os.path.join(noisy_dir, filename), cv2.IMREAD_GRAYSCALE)

            idx = save_random_crops(original_image, noisy_image, x_train_dir, y_train_dir, idx, k=image_size)

    print("Tüm parçalar kaydedildi.")
