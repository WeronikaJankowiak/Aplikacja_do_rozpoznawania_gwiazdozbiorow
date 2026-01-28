import csv
import math
from pathlib import Path

class StarCalculator:
    stars: dict[int, dict] | None = None
    constellations: dict[str, dict] | None = None

    def __init__(self):
        if StarCalculator.stars is None:
            StarCalculator.stars = self.LoadStarsFromCSV()
        if StarCalculator.constellations is None:
            StarCalculator.constellations = self.LoadConstellationsFromCSV()

    @staticmethod
    def LoadStarsFromCSV() -> dict[int, dict]:
        stars = {}
        stars_path = Path(__file__).resolve().parent / ".." / "a_extraction_generation_data" / "stars.csv"
        if not stars_path.exists():
            return {}

        with stars_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                star_id = int(row['star_id'])

                stars[star_id] = {
                    'star_id': star_id,
                    'star_name': row['star_name'],
                    'constellation_id': row['constellation_id'],
                    'ra': float(row['ra']), # RA (ang. right ascension) to odpowiednik długości geograficznej na sferze niebieskiej
                    'dec': float(row['dec']) # Dec (ang. declination) to odpowiednik szerokości geograficznej na sferze niebieskiej
                }

        return stars

    def GetStar(self, hip_id: int) -> dict | None:
        if StarCalculator.stars and hip_id in StarCalculator.stars:
            return StarCalculator.stars[hip_id]
        return None
    
    def GetStarName(self, hip_id: int) -> dict | None:
        if StarCalculator.stars and hip_id in StarCalculator.stars:
            return StarCalculator.stars[hip_id]['star_name']
        return ""

    @staticmethod
    def LoadConstellationsFromCSV() -> dict[str, dict]:
        constellations = {}
        constellation_path = Path(__file__).resolve().parent / ".." / "a_extraction_generation_data" / "constellations.csv"
        lines_path = Path(__file__).resolve().parent / ".." / "a_extraction_generation_data" / "constellation_lines.csv"

        with constellation_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                constellation_id = row['constellation_id']
                native_name = row['native_name']
                # english_name = row['english_name']
                polish_name = row['polish_name']
                constellations[constellation_id] = {
                    'constellation_id': constellation_id,
                    'constellation_name': polish_name or native_name,
                    'stars': [],
                    'lines': []
                }

        with lines_path.open(encoding="ansi", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                constellation_id = row['constellation_id']
                star1_hip = int(row['start_hip'])
                star2_hip = int(row['end_hip'])
                if constellation_id in constellations:
                    constellations[constellation_id]['lines'].append((star1_hip, star2_hip))

        return constellations

    def GetConstellationById(self, constellation_id: str) -> dict | None:
        if StarCalculator.constellations and constellation_id in StarCalculator.constellations:
            return StarCalculator.constellations[constellation_id]
        return None

    def GetConstellation(self, star_id: int) -> dict | None:
        star = self.GetStar(star_id)
        if star is None:
            return None
        constellation_id = star['constellation_id']
        return self.GetConstellationById(constellation_id)

    def GetConstellationLines(self, constellation_id: str) -> list[tuple[int, int]] | None:
        constellation = self.GetConstellationById(constellation_id)
        if constellation is None:
            return None
        return constellation['lines']

    def GetAngularSeparation(self, hip1: int, hip2: int) -> float | None:

        star1 = self.GetStar(hip1)
        star2 = self.GetStar(hip2)

        if star1 is None or star2 is None:
            return None

        ra1 = math.radians(star1['ra'])
        dec1 = math.radians(star1['dec'])
        ra2 = math.radians(star2['ra'])
        dec2 = math.radians(star2['dec'])

        # Obliczanie separacji kątowej za pomocą wzoru na sferyczną odległość (Sferyczne Prawo Cosinusów)
        sep = math.acos(math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra2 - ra1))

        return sep
    
    def GetAngular(self, hip1: int, hip2: int, hip3: int) -> float | None:
        star1 = self.GetStar(hip1)
        star2 = self.GetStar(hip2)
        star3 = self.GetStar(hip3)

        if star1 is None or star2 is None:
            return None

        ang12 = self.get_angular_separation(hip1, hip2)
        ang13 = self.get_angular_separation(hip1, hip3) 
        ang23 = self.get_angular_separation(hip2, hip3)

        # Dla uproszczenia obliczeń zakładamy, że ang12, ang13, ang23 są bokami płaskiego trójkąta
        if ang12 is None or ang13 is None or ang23 is None:
            return None
        # Zastosowanie wzoru cosinusów do obliczenia kąta przy wierzchołku star2
        cos_angle = (ang12**2 + ang23**2 - ang13**2) / (2 * ang12 * ang23)
        angle = math.acos(cos_angle)
        return angle

if __name__ == "__main__":
    calculator = StarCalculator()
    hip = 11767
    star = calculator.GetStar(hip)
    print(star)

    constellation_id = 'UMi'
    constellation = calculator.get_constellation_by_id(constellation_id)
    print(constellation)

    constellation = calculator.get_constellation(hip)
    print(constellation)

    lines = calculator.get_constellation_lines('UMi')
    print(lines)

    separation = calculator.get_angular_separation(11767, 11783)
    print(f"Separacja kątowa: {separation} radianów lub {math.degrees(separation)} stopni")
