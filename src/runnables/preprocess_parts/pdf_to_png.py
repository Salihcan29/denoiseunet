import os
from pdf2image import convert_from_path
from PIL import Image

"""
Bu çalıştırılabilir kod dosyası kaynak dizindeki tüm pdfleri png formatına çevirir

source: ./data/preprocess/original_pdfs/
target: ./data/preprocess/original_pngs/
"""


def pdf_to_png(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(source_dir, filename)

            images = convert_from_path(pdf_path)

            concatenated_image = Image.new('RGB', (images[0].width, images[0].height * len(images)))
            for i, image in enumerate(images):
                concatenated_image.paste(image, (0, i * images[0].height))

            png_filename = filename[:filename.rfind('.')] + '.png'
            png_path = os.path.join(target_dir, png_filename)

            concatenated_image.save(png_path, 'PNG')

    print("Tüm PDF dosyaları dönüştürüldü.")