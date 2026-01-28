import math
import time
import numpy as np
import pandas as pd

def RecognizeStars(sky_image, model, progress_callback=None):
    """
    Główna funkcja rozpoznawania gwiazd przy użyciu modelu ML.
    Analizuje regiony gwiazd na obrazie i identyfikuje je przy pomocy modelu uczenia maszynowego.
    """
    print("Rozpoznawanie gwiazd...")
    start_time = time.time()
    sky_image.break_recognition = False
    sky_image.recognized_stars = []
    sky_image.recognized_zeros = len(sky_image.star_regions)
    print("Rozpoznawanie gwiazd przy użyciu modelu ML...")
    if len(sky_image.star_regions) >= sky_image.star_numbers:
        total_regions = len(sky_image.star_regions)
        # Dla każdego regionu gwiazdy generujemy cechy i przewidujemy ID
        for index in range(total_regions):
            star = sky_image.star_regions[index]
            # Generowanie datasetu cech dla danej gwiazdy
            ds = pd.DataFrame(sky_image.GenerateDataset(index))
            x = ds.drop(columns=["star_id"])
            # Predykcja przy użyciu modelu ML
            pred = model.predict(x)
            predicted_ids, predicted_counts = np.unique(pred, return_counts=True)
            star["predicted_ids"] = list(predicted_ids)
            star["predicted_counts"] = list(predicted_counts)
            if progress_callback is not None:
                progress_callback("Rozpoznawanie gwiazdy przy użyciu modelu", index + 1, total_regions)
            if sky_image.break_recognition:
                return False

        recognized = []
        print("Analiza kombinacji gwiazd...")
        AnalyseCombinations(sky_image, recognized, 0, 0.0, progress_callback)

        # Przypisanie rozpoznanych ID do regionów gwiazd
        for idx, recognized_id in enumerate(sky_image.recognized_stars):
            sky_image.star_regions[idx]["predicted_star_id"] = recognized_id
            sky_image.star_regions[idx]["star_id"] = recognized_id

        num_recognized = sum(1 for value in sky_image.recognized_stars if value > 0)
        num_total = len(sky_image.star_regions)
        end_time = time.time()
        if progress_callback is not None:
            progress_callback("Znaleziono kombinację gwiazd", num_recognized, num_total)
        print(f"Czas rozpoznawania: {end_time - start_time:.2f} sekund")
        print(f"Rozpoznanych: {num_recognized}, łącznie: {num_total}")
        UpdateStarStatus(sky_image)
        return not sky_image.break_recognition

def AnalyseCombinations(sky_image, prev_recognized, level, confidence, progress_callback=None):
    """
    Rekurencyjna funkcja analizująca wszystkie możliwe kombinacje rozpoznanych gwiazd.
    Używa backtrackingu do znalezienia najlepszej kombinacji identyfikacji gwiazd.
    """
    if progress_callback is not None:
        progress_callback(
            "Znaleziono kombinację gwiazd",
            len(sky_image.star_regions) - sky_image.recognized_zeros,
            len(sky_image.star_regions),
        )
    # Pobieranie przewidywanych ID i liczności dla aktualnego poziomu
    ids = sky_image.star_regions[level]["predicted_ids"].copy()
    counts = sky_image.star_regions[level]["predicted_counts"].copy()
    if 0 not in ids:
        ids.append(0)
    if 0 not in counts:
        counts.append(0)
    recognized = prev_recognized.copy()
    recognized.append(0)
    # Obliczenie współczynników pewności dla każdego przewidywanego ID
    confidences = [float(value / sum(counts)) for value in counts]

    max_confidence = max(confidences)
    # Dynamiczne określenie minimalnego progu pewności
    min_confidence = max_confidence * max(min((len(sky_image.star_regions) - 5.0) / 20.0, 1.0), 0.0)

    if len(recognized) != level + 1:
        print("Błąd w RecognizeLevel")

    # Iteracja przez wszystkie możliwe ID gwiazd dla aktualnego poziomu
    for idx, star_id in enumerate(ids):
        star_id = int(star_id)
        star_confidence = confidences[idx]

        # Pomijanie ID z niskim poziomem pewności (chyba że id to 0 - gwiazda nierozpoznana)
        if star_confidence < min_confidence and star_id != 0:
            continue

        if sky_image.break_recognition:
            return

        sky_image.star_regions[level]["predicted_star_id"] = star_id
        sky_image.star_regions[level]["star_id"] = star_id

        # Sprawdzenie, czy ID nie jest już użyte (chyba że id to 0)
        if star_id not in recognized or star_id == 0:
            is_invalid = False
            recognized[level] = star_id

            # Sprawdzenie liczby nierozpoznanych gwiazd (0)
            number_of_zeros = sum(1 for value in recognized if value == 0)
            if number_of_zeros > sky_image.recognized_zeros:
                continue

            # Walidacja separacji kątowej między gwiazdami
            if level > 0 and star_id > 0:
                for inner in range(level):
                    if recognized[inner] > 0:
                        angular = sky_image.star_calculator.GetAngularSeparation(recognized[inner], star_id)
                        if angular > sky_image.max_angular_separation:
                            is_invalid = True
                            break

            # Walidacja geometryczna: sprawdzenie zgodności trójkątów
            if not is_invalid and level > 1 and star_id > 0:
                for s1 in range(level - 1):
                    if recognized[s1] > 0:
                        for s2 in range(s1 + 1, level):
                            if recognized[s2] > 0:
                                # Porównanie odległości na obrazie z rzeczywistymi kątowymi separacjami
                                star1 = sky_image.FindStarByHIP(recognized[s1])
                                star2 = sky_image.FindStarByHIP(recognized[s2])
                                star3 = sky_image.FindStarByHIP(star_id)
                                discrepancy = CalculateDistanceDiscrepancy(sky_image, star1, star2, star3)
                                if discrepancy > sky_image.max_error_triangle:
                                    is_invalid = True
                                    break
                                if sky_image.break_recognition:
                                    return
                        if is_invalid:
                            break

            if not is_invalid:
                if level < len(sky_image.star_regions) - 1:
                    AnalyseCombinations(sky_image, recognized, level + 1, confidence + star_confidence, progress_callback)
                else:
                    # Osiągnięto najgłębszy poziom - sprawdzenie, czy to najlepsza kombinacja
                    new_recognized_count = sum(1 for value in recognized if value > 0)
                    old_recognized_count = sum(1 for value in sky_image.recognized_stars if value > 0)
                    # Aktualizacja najlepszej kombinacji jeśli znaleziono lepszą
                    if new_recognized_count > old_recognized_count or (
                        new_recognized_count == old_recognized_count
                        and confidence + star_confidence > sky_image.recognized_confidence
                    ):
                        print(
                            f"Nowa najlepsza liczba rozpoznanych: {new_recognized_count} (stara: {old_recognized_count}), "
                            f"pewność: {confidence + star_confidence:.4f}"
                        )

                        sky_image.recognized_stars = recognized.copy()
                        sky_image.recognized_confidence = confidence + star_confidence
                        sky_image.recognized_zeros = level + 1 - new_recognized_count
                        if progress_callback is not None:
                            progress_callback("Znaleziono kombinację gwiazd", new_recognized_count, len(sky_image.star_regions))

def StatisticDistanceDiscrepancy(sky_image):
    """
    Oblicza statystyki rozbieżności odległości między trójkątami gwiazd.
    Używane do oceny dokładności rozpoznawania.
    """
    stat = [[] for _ in range(4)]
    print("Weryfikacja trójkątów...")

    # Iteracja przez wszystkie możliwe trójkąty gwiazd
    for s1 in range(len(sky_image.star_regions) - 2):
        star1 = sky_image.star_calculator.GetStar(sky_image.star_regions[s1]["predicted_star_id"])
        if star1 is not None and sky_image.star_regions[s1]["status"] in [sky_image.STAR_STATUS["OK"], sky_image.STAR_STATUS["Wrong"]]:
            for s2 in range(s1 + 1, len(sky_image.star_regions) - 1):
                star2 = sky_image.star_calculator.GetStar(sky_image.star_regions[s2]["predicted_star_id"])
                if star2 is not None and sky_image.star_regions[s2]["status"] in [sky_image.STAR_STATUS["OK"], sky_image.STAR_STATUS["Wrong"]]:
                    for s3 in range(s2 + 1, len(sky_image.star_regions)):
                        star3 = sky_image.star_calculator.GetStar(sky_image.star_regions[s3]["predicted_star_id"])
                        if star3 is not None and sky_image.star_regions[s3]["status"] in [sky_image.STAR_STATUS["OK"], sky_image.STAR_STATUS["Wrong"]]:
                            # Zliczanie poprawnie rozpoznanych gwiazd w trójkącie
                            num = 0
                            if sky_image.star_regions[s1]["status"] == sky_image.STAR_STATUS["OK"]:
                                num += 1
                            if sky_image.star_regions[s2]["status"] == sky_image.STAR_STATUS["OK"]:
                                num += 1
                            if sky_image.star_regions[s3]["status"] == sky_image.STAR_STATUS["OK"]:
                                num += 1

                            # Obliczenie rozbieżności odległości dla trójkąta
                            diff = CalculateDistanceDiscrepancy(sky_image, sky_image.star_regions[s1], sky_image.star_regions[s2], sky_image.star_regions[s3])
                            stat[num].append(diff)

    # Obliczenie średniej i odchylenia standardowego dla każdej grupy
    avg = []
    std_dev = []
    for bucket in stat:
        if bucket:
            values = np.array(bucket, dtype=float)
            avg.append(float(values.mean()))
            std_dev.append(float(values.std(ddof=0)))
        else:
            avg.append(0.0)
            std_dev.append(0.0)
    print(f"Średnia        : {[f'{value:.3f}' for value in avg]}")
    print(f"Odchylenie std.: {[f'{value:.3f}' for value in std_dev]}")
    return stat

def UpdateStarStatus(sky_image):
    """
    Aktualizuje status każdej gwiazdy na podstawie porównania oczekiwanego i przewidywanego ID.
    Statusy: Missed (pominięta), OK (poprawna), Wrong (błędna), Recognized (rozpoznana), Unknown (nieznana)
    """
    for star in sky_image.star_regions:
        if star["expected_star_id"] > 0 and star["predicted_star_id"] == 0:
            star["status"] = sky_image.STAR_STATUS["Missed"]
        elif star["expected_star_id"] > 0 and star["predicted_star_id"] > 0 and star["predicted_star_id"] == star["expected_star_id"]:
            star["status"] = sky_image.STAR_STATUS["OK"]
        elif star["expected_star_id"] > 0 and star["predicted_star_id"] > 0 and star["predicted_star_id"] != star["expected_star_id"]:
            star["status"] = sky_image.STAR_STATUS["Wrong"]
        elif star["expected_star_id"] == 0 and star["predicted_star_id"] > 0:
            star["status"] = sky_image.STAR_STATUS["Recognized"]
        else:
            star["status"] = sky_image.STAR_STATUS["Unknown"]

def CalculateDistanceDiscrepancy(sky_image, star1, star2, star3):
    """
    Oblicza rozbieżność między obserwowanymi odległościami na obrazie a rzeczywistymi
    separacjami kątowymi dla trójkąta utworzonego przez trzy gwiazdy.
    Zwraca sumę bezwzględnych różnic znormalizowanych odległości.
    """
    # Obliczenie odległości na obrazie między gwiazdami
    distance12 = sky_image.DistanceBetweenStars(star1, star2)
    distance13 = sky_image.DistanceBetweenStars(star1, star3)
    distance23 = sky_image.DistanceBetweenStars(star2, star3)
    # Normalizacja odległości względem obwodu trójkąta
    perimeter = distance12 + distance13 + distance23
    distance12 /= perimeter
    distance13 /= perimeter
    distance23 /= perimeter

    # Obliczenie rzeczywistych separacji kątowych z katalogu
    angle12 = sky_image.star_calculator.GetAngularSeparation(star1["predicted_star_id"], star2["predicted_star_id"])
    angle13 = sky_image.star_calculator.GetAngularSeparation(star1["predicted_star_id"], star3["predicted_star_id"])
    angle23 = sky_image.star_calculator.GetAngularSeparation(star2["predicted_star_id"], star3["predicted_star_id"])
    # Normalizacja kątów względem sumy
    angle_sum = angle12 + angle13 + angle23
    angle12 /= angle_sum
    angle13 /= angle_sum
    angle23 /= angle_sum

    # Obliczenie sumy bezwzględnych różnic
    diff = math.fabs(distance12 - angle12)
    diff += math.fabs(distance13 - angle13)
    diff += math.fabs(distance23 - angle23)
    return diff
