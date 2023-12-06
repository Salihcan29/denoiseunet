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


def run(crops_per_pdf=1000, image_size=64):
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
            temp = noise_generator.addPixels(image, blackProb=0.12, whiteProb=0.12)
            noisy_image_path = os.path.join(target_dir, filename[:-4]+'_addPixels.png')
            cv2.imwrite(noisy_image_path, temp)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            temp = noise_generator.addSpeckleNoise(image, speckle_intensity=0.4)
            noisy_image_path = os.path.join(target_dir, filename[:-4] + '_speckleIntensity.png')
            cv2.imwrite(noisy_image_path, temp)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            temp = noise_generator.makeGrayNoiseAreas(image, noise_prob=0.4)
            noisy_image_path = os.path.join(target_dir, filename[:-4] + '_grayNoiseAreas.png')
            cv2.imwrite(noisy_image_path, temp)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            temp = noise_generator.addRandomLines(image, line_num=500, line_thickness=1, line_color=192)
            noisy_image_path = os.path.join(target_dir, filename[:-4] + '_addRandomLines.png')
            cv2.imwrite(noisy_image_path, temp)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            temp = noise_generator.addBlur(image, blur_strength=0.5)
            noisy_image_path = os.path.join(target_dir, filename[:-4] + '_addBlur.png')
            cv2.imwrite(noisy_image_path, temp)


    print("Tüm resim dosyalarına gürültü eklendi.")

    original_dir = './data/preprocess/original_pngs/'
    noisy_dir = './data/preprocess/noisy_pngs/'

    y_train_dir = './data/dataset/y_train/'
    x_train_dir = './data/dataset/x_train/'
    idx = 0

    functions = ['addPixels',
                 'speckleIntensity',
                 'grayNoiseAreas',
                 'addRandomLines',
                 'addBlur']

    for filename in os.listdir(original_dir):
        if filename.endswith(".png"):
            original_image = cv2.imread(os.path.join(original_dir, filename), cv2.IMREAD_GRAYSCALE)

            for function in functions:
                noisy_image = cv2.imread(os.path.join(noisy_dir, f'{filename[:-4]}_{function}.png'), cv2.IMREAD_GRAYSCALE)
                idx = save_random_crops(original_image,
                                        noisy_image,
                                        x_train_dir,
                                        y_train_dir,
                                        idx,
                                        n_crops=crops_per_pdf,
                                        k=image_size)

    print("Tüm parçalar kaydedildi.")
