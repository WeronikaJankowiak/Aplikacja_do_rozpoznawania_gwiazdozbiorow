# pip install astropy astroquery
import csv
import json
from astroquery.vizier import Vizier
from pathlib import Path


class SourceData:
    """Klasa do przygotowywania danych źródłowych o gwiazdach i gwiazdozbiorach."""

    dict_star_names: dict[int, str] | None = None
    translation_dict: dict[str, str] | None = None

    def __init__(self) -> None:
        # Inicjalizacja wspólnego słownika nazw gwiazd przy pierwszym utworzeniu instancji
        if SourceData.dict_star_names is None:
            SourceData.dict_star_names = self.LoadStarNames()
        if SourceData.translation_dict is None:
            SourceData.translation_dict = self.LoadTranslationDict()

    def LoadStarNames(self) -> dict[int, str]:
        """Wczytuje nazwy gwiazd z pliku CSV i zwraca mapowanie HIP -> nazwa."""
        csv_path = Path(__file__).resolve().parent / ".." / "resources" / "star_names.csv"
        print(f"Loading star names from: {csv_path}")
        if not csv_path.exists():
            return {}

        lookup: dict[int, str] = {}
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        hip = int(row.get("star_id", ""))
                    except (TypeError, ValueError):
                        continue
                    name = (row.get("star_name") or "").strip()
                    if name:
                        lookup[hip] = name
        except Exception:
            return {}
        return lookup

    def LoadTranslationDict(self) -> dict[str, str]:
        """Wczytuje słownik tłumaczeń nazw z pliku CSV i zwraca mapowanie nazwa_oryginalna -> nazwa_przetłumaczona."""
        csv_path = Path(__file__).resolve().parent / ".." / "resources" / "translations.csv"
        print(f"Loading translations from: {csv_path}")
        if not csv_path.exists():
            return {}

        lookup: dict[str, str] = {}
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    original_name = (row.get("native_name") or "").strip()
                    translated_name = (row.get("polish_name") or "").strip()
                    if original_name and translated_name:
                        lookup[original_name] = translated_name
        except Exception:
            return {}
        return lookup

    def PrepareSourceData(self):
        stars = {}
        constellations = []
        const_lines = []

        json_path = Path(__file__).resolve().parent / ".." / "resources" / "index.json"
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        const_source = data.get("constellations") if isinstance(data, dict) else None

        for const in const_source:
            # dodanie gwiazdozbioru
            const_id = const.get("id")[11:]
            const_native = const["common_name"].get("native", "")
            const_english = const["common_name"].get("english", "")
            const_polish =  SourceData.Translate(const_native)

            constellations.append({
                "constellation_id": const_id,
                "native_name": const_native,
                "english_name": const_english,
                "polish_name": const_polish
            })
            print(f"Constellation: {const_id} - {const_native} / {const_english} / {const_polish}")

            # dodanie linii gwiazdozbioru
            for line in const.get("lines", []):
                points: list[int] = []
                for point in line:
                    if type(point) != int:
                        break
                    points.append(int(point))
                    if point not in stars:
                        stars[point] = {
                            "star_id": int(point),
                            "constellation_id": const_id,
                        }
                    else:
                        stars[point]["constellation_id"] = const_id

                for start, end in zip(points, points[1:]):
                    const_lines.append({
                        "constellation_id": const_id,
                        "start_hip": start,
                        "end_hip": end
                    })

        # dodanie brakujących gwiazd
        for star_id in self.dict_star_names.keys():
            if star_id not in stars:
                stars[star_id] = {
                    "star_id": star_id,
                    "star_name": self.dict_star_names[star_id]
                }

        # uzupełnienie danych o gwiazdach
        for hip_id in stars.keys():
            # hip_id = 85258
            Vizier.ROW_LIMIT = 1
            # cols = ["Star name", "RArad", "DErad", "Constellation"]
            cols = ["**"]
            res = Vizier(columns=cols).query_constraints(catalog="I/311/hip2", HIP=str(hip_id))
            if len(res) > 0:
                row = res[0][0]
                # print(res)
                stars[hip_id]["ra"] = row["RArad"]
                stars[hip_id]["dec"] = row["DErad"]
                if "constellation_id" not in stars[hip_id] or stars[hip_id]["constellation_id"] == "": 
                    stars[hip_id]["constellation_id"] = ""
                    if "Constellation" in row.colnames and row["Constellation"]:
                        stars[hip_id]["constellation_id"] = row["Constellation"][:3]
                if "Star name" in row.colnames:
                    stars[hip_id]["star_name"] = row["Star name"]
                else:
                    stars[hip_id]["star_name"] = self.dict_star_names.get(hip_id, "")

            print(stars[hip_id])
            # print(f"Star HIP {hip_id}: {stars[hip_id]['star_name']} RA: {stars[hip_id]['ra']} DEC: {stars[hip_id]['dec']}")

        self.SaveToCSV(constellations, const_lines, stars)

    def Translate(native_name: str) -> str:
        polish_name = SourceData.translation_dict.get(native_name,"")
        return polish_name

    def SaveToCSV(self, constellations, const_lines, stars):
        with open("./a_extraction_generation_data/constellations.csv", 'w', newline='', encoding="utf-8") as csvfile:
            fieldsC = ["constellation_id", "native_name", "english_name", "polish_name"]
            writerC = csv.DictWriter(csvfile, fieldnames=fieldsC)
            writerC.writeheader()
            writerC.writerows(constellations)

        with open("./a_extraction_generation_data/constellation_lines.csv", 'w', newline='', encoding="utf-8") as csvfile:
            fieldsCL = ["constellation_id", "start_hip", "end_hip"]
            writerCL = csv.DictWriter(csvfile, fieldnames=fieldsCL)
            writerCL.writeheader()
            writerCL.writerows(const_lines)

        with open("./a_extraction_generation_data/stars.csv", 'w', newline='', encoding="utf-8") as csvfile:
            fieldsS = ["star_id", "star_name", "constellation_id", "ra", "dec"]
            writerS = csv.DictWriter(csvfile, fieldnames=fieldsS)
            writerS.writeheader()
            writerS.writerows(stars.values())

if __name__ == "__main__":
    source_data = SourceData()
    source_data.PrepareSourceData()