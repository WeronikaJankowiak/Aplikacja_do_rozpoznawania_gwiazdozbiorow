import time
import numpy as np
import os
import glob
from PIL import Image as im
from scipy import ndimage


def load_image(file_path):
    """Wczytuje obraz i zamienia na czarno-białe."""
    image = im.open(file_path).convert('L')
    return image


def gaussian_highpass_filter(image, sigma=3):
    """
    Filtr górno-przepustowy Gaussa - odejmowanie rozmycia gaussowskiego od oryginału.
    """
    time0 = time.time()
    data = np.array(image, dtype=float)
    lowpass = ndimage.gaussian_filter(data, sigma)
    highpass = data - lowpass
    time1 = time.time()
    elapsed = time1 - time0
    return im.fromarray(highpass).convert('L'), np.array(highpass), elapsed


def laplacian_filter(image):
    """
    Filtr Laplace'a (Wykrywa krawędzie i punkty poprzez obliczanie drugiej pochodnej.)
    """
    time0 = time.time()
    data = np.array(image, dtype=float)
    laplacian = ndimage.laplace(data)
    # Inversion i normalizacja
    laplacian = -laplacian
    laplacian = np.clip(laplacian, 0, 255)
    time1 = time.time()
    elapsed = time1 - time0
    return im.fromarray(laplacian).convert('L'), laplacian, elapsed


def tophat_filter(image, size=5):
    """
    Morfologiczny filtr top-hat.
    Rozmiar elementu strukturalnego: (3 + 2*size) × (3 + 2*size) pikseli
    Przykład: size=1 → 5×5, size=2 → 7×7, size=3 → 9×9
    """
    time0 = time.time()
    data = np.array(image, dtype=np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)
    for i in range(size):
        struct = ndimage.binary_dilation(struct)
    tophat = ndimage.white_tophat(data, structure=struct)
    time1 = time.time()
    elapsed = time1 - time0
    return im.fromarray(tophat), tophat, elapsed


def median_subtraction_filter(image, size=5):
    """
    Mediana z odejmowaniem.
    """
    time0 = time.time()
    data = np.array(image, dtype=float)
    median = ndimage.median_filter(data, size=size)
    result = data - median
    result = np.clip(result, 0, 255)
    time1 = time.time()
    elapsed = time1 - time0
    return im.fromarray(result).convert('L'), result, elapsed


def save_image(image, output_path):
    """Zapisuje obraz do pliku"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def test_all_filters_on_images():
    """
    Testuje wszystkie filtry z różnymi parametrami na wszystkich obrazach z folderu.
    """
    # Ścieżka do folderu wewnątrz modułu b_image_processing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "filter_test_photos")
    output_folder = os.path.join(input_folder, "results_test") # aby nadpisać results usunąć _test
    
    # Tworzenie folderu wynikowego
    os.makedirs(output_folder, exist_ok=True)
    
    # Wczytanie listy plików obrazów
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    # Usunięcie duplikatów (Windows jest case-insensitive)
    image_files = list(set(image_files))
    image_files.sort()  # Sortowanie dla powtarzalności
    
    if not image_files:
        print(f"BŁĄD: Nie znaleziono obrazów w folderze '{input_folder}'")
        print(f"Upewnij się, że folder istnieje i zawiera pliki obrazów (.jpg, .png, itp.)")
        return
    
    # Ograniczenie do 10 plików
    image_files = image_files[:10]
    
    print("="*70)
    print("PORÓWNANIE FILTRÓW DO DETEKCJI GWIAZD")
    print("="*70)
    print(f"\nFolder wejściowy: {input_folder}")
    print(f"Folder wyjściowy: {output_folder}")
    print(f"Liczba obrazów do przetworzenia: {len(image_files)}\n")
    
    # Definicja testów dla każdego filtra z różnymi parametrami
    filter_tests = [
        # Gaussian High-Pass z różnymi sigma
        ("Gaussian_HighPass_sigma1", lambda img: gaussian_highpass_filter(img, sigma=1)),
        ("Gaussian_HighPass_sigma2", lambda img: gaussian_highpass_filter(img, sigma=2)),
        ("Gaussian_HighPass_sigma3", lambda img: gaussian_highpass_filter(img, sigma=3)),
        ("Gaussian_HighPass_sigma5", lambda img: gaussian_highpass_filter(img, sigma=5)),
        
        # Laplacian (bez parametrów)
        ("Laplacian", lambda img: laplacian_filter(img)),
        
        # Top-Hat z różnymi rozmiarami
        ("TopHat_size3", lambda img: tophat_filter(img, size=0)),  # okno 3×3
        ("TopHat_size5", lambda img: tophat_filter(img, size=1)),  # okno 5×5
        ("TopHat_size7", lambda img: tophat_filter(img, size=2)),  # okno 7×7
        ("TopHat_size9", lambda img: tophat_filter(img, size=3)),  # okno 9×9
        
        # Median Subtraction z różnymi rozmiarami
        ("Median_size3", lambda img: median_subtraction_filter(img, size=3)),
        ("Median_size5", lambda img: median_subtraction_filter(img, size=5)),
        ("Median_size7", lambda img: median_subtraction_filter(img, size=7)),
        ("Median_size9", lambda img: median_subtraction_filter(img, size=9)),
    ]
    
    total_tests = len(image_files) * len(filter_tests)
    current_test = 0
    
    # Słownik do zbierania statystyk czasów dla każdej konfiguracji
    filter_stats = {filter_name: [] for filter_name, _ in filter_tests}
    
    # Przetwarzanie każdego obrazu
    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        print("-"*70)
        print(f"Przetwarzanie: {os.path.basename(img_path)}")
        print("-"*70)
        
        try:
            # Wczytanie obrazu
            source_image = load_image(img_path)
            
            # Zapis oryginału
            original_path = os.path.join(output_folder, f"{img_name}_00_original.png")
            save_image(source_image, original_path)
            
            # Testowanie każdego filtra
            for filter_name, filter_func in filter_tests:
                current_test += 1
                try:
                    filtered_image, filtered_bitmap, elapsed_time = filter_func(source_image)
                    
                    # Zapisanie czasu wykonania
                    filter_stats[filter_name].append(elapsed_time)
                    
                    output_filename = f"{img_name}_{filter_name}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    save_image(filtered_image, output_path)
                    
                    progress = (current_test / total_tests) * 100
                    print(f"  [{progress:5.1f}%] {filter_name:30s} - {elapsed_time:.4f}s")
                    
                except Exception as e:
                    print(f"  [BŁĄD] {filter_name}: {e}")
            
            print()
            
        except Exception as e:
            print(f"  BŁĄD przy wczytywaniu {img_path}: {e}\n")
            continue
    
    print("="*70)
    print("ZAKOŃCZONO TESTOWANIE")
    print("="*70)
    print(f"Przetworzone obrazy: {len(image_files)}")
    print(f"Zastosowane filtry: {len(filter_tests)}")
    print(f"Wyniki zapisane w: {output_folder}")
    print("="*70)
    
    # Wyświetlenie statystyk czasów wykonania
    print("\nSTATYSTYKI CZASÓW WYKONANIA:")
    print("="*70)
    print(f"{'Filtr':<30} {'Średni [s]':<12} {'Min [s]':<12} {'Max [s]':<12}")
    print("-"*70)
    
    for filter_name, _ in filter_tests:
        times = filter_stats[filter_name]
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{filter_name:<30} {avg_time:<12.4f} {min_time:<12.4f} {max_time:<12.4f}")
        else:
            print(f"{filter_name:<30} {'Brak danych':<12}")
    
    print("="*70)


if __name__ == "__main__":
    test_all_filters_on_images()
    

