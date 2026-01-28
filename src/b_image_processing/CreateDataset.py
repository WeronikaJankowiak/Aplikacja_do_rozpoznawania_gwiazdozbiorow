import csv
import os

from pathlib import Path

from SkyImage import SkyImage


# WAŻNE! aby działało trzeba przestawić ścieżkę w skyimage
dataset_fields = ["star_id", "R1", "R2", "R3", "L1", "L2"]
dataset_path = './b_image_processing/Dataset_test.csv' # aby nadpisać któryś z datasetów trzeba tu wstawić jego nazwę
source_folder = Path(__file__).resolve().parent / ".." / "a_extraction_generation_data" / "Screenshots_1000"
extension_image = '.png'


def Analyse_images():
    # Pobranie listy plików i folderów w podanej ścieżce
    files_and_folders = os.listdir(source_folder)
    star_numbers = 4  # liczba gwiazd do analizy

    # utworzenie nowego pliku do zapisu
    with open(dataset_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dataset_fields)
        writer.writeheader()

    # Analizowanie tylko plików o określonym rozszerzeniu
    for file_name in files_and_folders:
        if os.path.isfile(os.path.join(source_folder, file_name)) and file_name.endswith(extension_image):
            print(file_name)

            # utworzenie obiektu SkyImage
            image = SkyImage(source_folder, file_name, extension_image)
            print(f"Liczba wykrytych gwiazd: {len(image.star_regions)}")

            # zapis dataset do pliku
            with open(dataset_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dataset_fields)
                dataset = image.GetDataset(star_numbers)

                writer.writerows(dataset)

if __name__ == "__main__":
    Analyse_images()