import math
import time
import numpy as np
import os
import pandas as pd
import re

from PIL import Image as im
from scipy import ndimage

from b_image_processing.StarCalculation import StarCalculator # dla wywołania z main
# from StarCalculation import StarCalculator # dla wywołania z CreateDataset

class SkyImage:
    source_image = []
    filtered_image = []
    filtered_bitmap = []
    file_extension = ''
    file_name = ''
    file_path = ''
    source_csv = pd.DataFrame()
    star_calculator = StarCalculator()
    threshold = 80
    star_numbers = 4
    rows = 0
    columns = 0


    STAR_STATUS = {
        "Unknown": 0,
        "Missed": 1,        # gwiazda nie została rozpoznana, ale jest w pliku csv
        "OK": 2,            # gwiazda została poprawnie rozpoznana
        "Wrong": 3,         # gwiazda została rozpoznana, ale błędnie
        "Recognized": 4,    # gwiazda została rozpoznana, ale nie ma pliku csv żeby to sprwdzić
        "Calculated": 5,    # gwiazda została obliczona na podstawie innych gwiazd ale nie ma jej na zdjęciu
        "Assigned": 6,      # gwiazda została obliczona na podstawie innych gwiazd i już była na zdjęciu
        "Outside": 7        # gwiazda jest poza obszarem zdjęcia
    }

    def __init__(self, folder, file_name, file_extension = ''):
        self.file_name = file_name
        self.file_extension = file_extension
        self.file_path = os.path.join(folder, file_name)
        self.source_csv = pd.DataFrame()
        self.star_regions = []
        self.constellations = {}
        self.is_recognized = False
        self.distances = {}
        self.angles = {}
        self.recognized_stars = []
        self.recognized_confidence = 0.0
        self.recognized_zeros = 0
        self.max_luminosity = 0
        self.max_error_triangle = 0.05
        self.max_error_distance = 0.2
        self.max_angular_separation = math.radians(90.0)    
        self.star_statistic = {}
        self.break_recognition = False
        
        # otwarcie zdjęcia i zamiana na czarnobiałe
        time0 = time.time()
        self.source_image = im.open(self.file_path).convert('L')
        time1 = time.time()
        print(f"Czas wczytywania obrazu: {time1 - time0:.2f} sekund")

        # od oryginalnego obrazka odejmowane jest rozmycie gaussowskie (filtr dolnoprzepustowy) i powstaje filtr górno-przepustowy
        data = np.array(self.source_image, dtype=float)
        lowpass = ndimage.gaussian_filter(data, 3)
        gauss_highpass = data - lowpass
        time2 = time.time()
        print(f"Czas filtrowania obrazu: {time2 - time1:.2f} sekund")
        self.filtered_image = im.fromarray(gauss_highpass).convert('L')
        time3 = time.time()
        print(f"Czas konwersji obrazu: {time3 - time2:.2f} sekund")
        # self.filtered_image.save('filtered.png')
        time4 = time.time()
        # print(f"Filtered image saving time: {time4 - time3:.2f} seconds")
        self.filtered_bitmap = np.array(self.filtered_image)
        time5 = time.time()
        print(f"Czas tworzenia bitmapy: {time5 - time4:.2f} sekund")

        # jeżeli istniej plik csv z listą występujących gwiazd to otwiera go
        if (file_extension != ''):
            self.Read_source_CSV()
        time6 = time.time()
        print(f"Czas odczytu CSV: {time6 - time5:.2f} sekund")

        self.Search_Stars()
        self.threshold = 50 * self.max_luminosity / 255
        print(f"   Próg obrazu ustawiony na {self.threshold:.2f} (max: {self.max_luminosity:.2f})")
        time7 = time.time()
        print(f"Czas wyszukiwania gwiazd: {time7 - time6:.2f} sekund")

        self.CalculateDistancesAndAngles()
        time8 = time.time()
        print(f"Czas obliczania odległości i kątów: {time8 - time7:.2f} sekund")
        print(f"Całkowity czas przetwarzania obrazu: {time8 - time0:.2f} sekund")

    def Read_source_CSV(self):
        # funkcja odczytująca plik csv
        # plik csv ma taką samą nazwę jak zdjęcie
        csv_path = re.sub(self.file_extension, '.csv', self.file_path)
        # sprawdzanie czy plik istnieje
        if (os.path.exists(csv_path)):
            # odczytanie pliku
            self.source_csv = pd.read_csv(csv_path, encoding='ansi')
            # print(self.source_csv)
        # print(csv_path)

    def Scan_Region(self, minX, maxX, minY, maxY):
        # sprawdza czy nie bada obszaru poza zdjęciem
        if (minX < 0) or (minY < 0) or (maxX >= self.columns) or (maxY >= self.rows):
            return False

        # funkcja sprawdza czy na ustalonym obszarze bitmapy nie występują piksele o jasności przekraczającej ustalony próg
        for y in range(minY, min(maxY+1, self.rows)):
            for x in range(minX, min(maxX+1, self.columns)):
                if (self.filtered_bitmap[y][x] > self.threshold):
                    return True
        return False

    def Search_Stars(self):
        # funkcja wyszukująca obiekty (gwiazdy, planety, ...) na zdjęciu
        self.rows = len(self.filtered_bitmap)
        self.columns = len(self.filtered_bitmap[0])

        # przeszukanie całego obszaru zdjęcia
        for y in range(self.rows):
            for x in range(self.columns):
                # ustawienie maksymalnej jasności
                if int(self.filtered_bitmap[y][x]) > self.max_luminosity:
                    self.max_luminosity = int(self.filtered_bitmap[y][x])

                # sprawdzenie czy piksel nie znajduje się w poprzednio znalezionych obszarach
                found_region = False
                for region in self.star_regions:
                    if (region['minX'] <= x) and (region['maxX'] >= x) and (region['minY'] <= y) and (region['maxY'] >= y):
                        found_region = True
                        break

                if not found_region:
                    # jeśli jasność piksela przekracza próg (threshold) zaczynamy badać ten obszar
                    if (self.filtered_bitmap[y][x] > self.threshold):
                        found_piksel = True
                        minX = x
                        maxX = x
                        minY = y
                        maxY = y
                        while found_piksel:
                            found_piksel = self.Scan_Region(minX, maxX, minY-3, minY-1) # skanowanie wiersza powyżej
                            if found_piksel:
                                minY -= 3
                            else:
                                found_piksel = self.Scan_Region(minX, maxX, maxY+1, maxY+3) # skanowanie wiersza poniżej
                                if found_piksel:
                                    maxY += 3
                                else:
                                    found_piksel = self.Scan_Region(minX-3, minX-1, minY, maxY) # skanowanie kolumny z lewej
                                    if found_piksel:
                                        minX -= 3
                                    else:
                                        found_piksel = self.Scan_Region(maxX+1, maxX+3, minY, maxY) # skanowanie kolumny z prawej
                                        if found_piksel:
                                            maxX += 3

                        # powiększenie obszaru o 1 piksel z każdej strony. Bez tego nie wszystkie gwiazdy z pliku csv są odnalezione.
                        if minX > 0:
                            minX -= 1
                        if maxX < self.columns - 1:
                            maxX += 1
                        if minY > 0:
                            minY -= 1
                        if maxY < self.rows - 1:
                            maxY += 1

                        # wyszukanie gwiazdy w csv
                        star_id = 0
                        for index, row in self.source_csv.iterrows():
                            if row['X'] >= minX and row['X'] <= maxX and row['Y'] >= minY and row['Y'] <= maxY:
                                star_id = row['Star_ID']

                        # dodanie regionu do listy
                        new_region = {
                            "star_id": 0,
                            "expected_star_id": star_id,
                            "predicted_star_id": 0,
                            "calculated_star_id": 0,
                            "confidence": 0.0,
                            "minX": minX,
                            "maxX": maxX,
                            "minY": minY,
                            "maxY": maxY,
                            "X": float(maxX + minX) / 2.0,
                            "Y": float(maxY + minY) / 2.0,
                            "status": self.STAR_STATUS['Unknown'],
                        }
                        # print(new_region)
                        self.star_regions.append(new_region)

    def GetDataset(self, star_numbers=None):
        # pobranie datasetu
        dataset = []

        if star_numbers is not None:
            self.star_numbers = star_numbers

        for s1 in range(len(self.star_regions)):
            if self.star_regions[s1]['expected_star_id'] != 0:
                dataset += self.GenerateDataset(s1)
        return dataset

    def GenerateDataset(self, s1):
        # generowanie datasetu dla 4 gwiazd
        dataset = []

        for s2 in range(len(self.star_regions)-self.star_numbers+2):
            if s2 != s1:
                for s3 in range(s2+1, len(self.star_regions)-self.star_numbers+3):
                    if s3 != s1:
                        for s4 in range(s3+1, len(self.star_regions)-self.star_numbers+4):
                            if s4 != s1:
                                # Generowanie datasetu dla 4 gwiazd
                                ang = {}
                                ang[(self.angles[s1, s2],s2)] = s2
                                ang[(self.angles[s1, s3],s3)] = s3
                                ang[(self.angles[s1, s4],s4)] = s4
                                sorted_angles = sorted(ang, key=lambda item: item[0])
                                
                                st2 = sorted_angles[0][1]
                                st3 = sorted_angles[1][1]
                                st4 = sorted_angles[2][1]
                                
                                dist_R = []
                                dist_L = []
                                dist_R.append(self.distances[s1, st2])
                                dist_R.append(self.distances[s1, st3])
                                dist_R.append(self.distances[s1, st4])
                                dist_L.append(self.distances[st2, st3])
                                dist_L.append(self.distances[st3, st4])
                                dist_L.append(self.distances[st4, st2])
                                
                                total_sum = sum(dist_R) + sum(dist_L)
                                dist_R = [d / total_sum for d in dist_R]
                                dist_L = [d / total_sum for d in dist_L]
                                
                                start = int(np.argmax(dist_R))
                                target = {'star_id': self.star_regions[s1]['expected_star_id']}
                                target['R1'] = dist_R[(0 + start) % 3]
                                target['R2'] = dist_R[(1 + start) % 3]
                                target['R3'] = dist_R[(2 + start) % 3]
                                target['L1'] = dist_L[(0 + start) % 3]
                                target['L2'] = dist_L[(1 + start) % 3]
                                dataset.append(target)

        return dataset

    def DistanceBetweenStars(self, star1, star2):

        dx = star1['X'] - star2['X']
        dy = star1['Y'] - star2['Y']
        return math.sqrt(dx * dx + dy * dy)

    def AngleBetweenStars(self, star1, star2):
        dx = star2['X'] - star1['X']
        dy = star2['Y'] - star1['Y']

        if dx != 0.0:
            angle = math.atan2(dy, dx)
            if dx < 0:
                angle += math.pi
        else:
            angle = math.pi / 2.0
            if dy < 0:
                angle += math.pi

        if angle < 0:
            angle += 2 * math.pi

        return angle

    def FindStarByHIP(self, hip_id: int):
        for star in self.star_regions:
            if star['star_id'] == hip_id:
                return star
        return None

    def CalculateDistancesAndAngles(self):
        self.distances = {}
        self.angles = {}
        for s1 in range(len(self.star_regions)-1):
            star_1 = self.star_regions[s1]
            for s2 in range(s1+1, len(self.star_regions)):
                star_2 = self.star_regions[s2]
                distance = self.DistanceBetweenStars(star_1, star_2)
                angle = self.AngleBetweenStars(star_1, star_2)
                self.distances[s1, s2] = distance
                self.distances[s2, s1] = distance
                self.angles[s1, s2] = angle
                if angle < math.pi:
                    self.angles[s2, s1] = angle + math.pi
                else:
                    self.angles[s2, s1] = angle - math.pi

