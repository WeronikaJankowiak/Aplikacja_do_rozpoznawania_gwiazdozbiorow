# Aplikacja do rozpoznawania gwiazdozbiorów

Projekt służy do rozpoznawania gwiazdozbiorów i ich gwiazd na zdjęciach nieba. Aplikacja zawiera pipeline przygotowania danych, przetwarzania obrazów, uczenia modeli oraz interfejs graficzny (PySide6).

## Struktura aplikacji
```
src
├── main.py                          # Główny plik programu uruchamia aplikację z interfejsem
├── a_extraction_generation_data     # Przygotowanie danych źródłowych (CSV) i par obrazów i metadanych
│   ├── PrepareSourceData.py
│   ├── constellations.csv
│   ├── constellation_lines.csv
│   ├── stars.csv
│   ├── _Stars_new.ssc               # Ponowne generowanie obrazów wymaga instalacji Stellarium
│   ├── Screenshots_1000             # 1000 par obraz + metadane
│   │   ├── sky_030000.csv
│   │   ├── sky_030000.png
│   │   └── ...
│   └── Screenshots_5000             # 5000 par obraz + metadane
│       ├── sky_050000.csv
│       ├── sky_050000.png
│       └── ...
├── b_image_processing               # Przetwarzanie obrazów i porównanie filtrów
│   ├── CreateDataset.py
│   ├── Dataset_1000.csv
│   ├── Dataset_5000.csv
│   ├── FilterTest.py
│   ├── SkyImage.py
│   ├── StarCalculation.py
│   └── filter_test_photos
│       ├── [10 zdjęć testowych]
│       ├── results/                
│       │   └── [wyniki testu filtrów na 10 zdjęciach]
│       └── results
├── c_stars_recognizing              # Uczenie algorytmów uczenia maszynowego i rozpoznawanie gwiazd
│   ├── LearnModel.py
│   └── RecognizeStars.py
├── d_constellations_recognizing     # Rozpoznawanie gwiazdozbiorów
│   └── RecognizeConstellations.py
├── e_ui                             # Interfejs graficzny
│   ├── __init__.py
│   └── main_window.py
├── resources                        # Pliki danych pobrane ze Stellarium
│   ├── __init__.py
│   ├── index.json
│   ├── star_names.csv
│   └── translations.csv
└── widgets                          # Wymagane do działania Pyside6
    └── __init__.py
```

## WAŻNE: INSTALACJA I URUCHOMIENIE
```
Aby zainstalować, a następnie uruchomić aplikację należy:
• utworzyć na dysku lokalnym C folder _Gwiazdozbiory,
• rozpakować plik ZIP do nowo utworzonego folderu,
• pobrać Python w wersji 3.14.2 (https://www.python.org/downloads/release/python-3142/).
Aplikacja była pisana na wersji 3.10 ale przetestowano jej działanie dla wersji 3.14.2
i aplikacja działała prawidłowo. Wybór wersji wynika z dedykowanego instalatora na
system Windows, którego wersja 3.10 nie posiada. Podczas instalacji należy zaznaczyć
opcję „Add Python to PATH”.
• otworzyć „Wiersz polecenia” w systemie Windows,
• w wywołanej konsoli wpisać kolejno komendy (każdą zatwierdzić przyciskiem „Enter”):
– cd C:\_Gwiazdozbiory\Aplikacja,
– python -m venv gwiazdozbiory_env,
– gwiazdozbiory_env\Scripts\activate,
– pip install -r requirements.txt,
– python src\main.py.

```
