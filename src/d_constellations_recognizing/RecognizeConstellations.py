import math


def RecognizeConstellations(sky_image, model, progress_callback=None):
    """
    Rozpoznaje gwiazdozbiory na podstawie zidentyfikowanych gwiazd.
    Łączy rozpoznane gwiazdy z ich gwiazdozbiorami i oblicza brakujące gwiazdy w liniach gwiazdozbiorów.
    """
    print("Rozpoznawanie gwiazdozbiorów...")
    # Filtrowanie gwiazd, które zostały rozpoznane (mają przypisane ID)
    predicted_stars = [star for star in sky_image.star_regions if star["predicted_star_id"] != 0]
    total_predicted = len(predicted_stars)
    processed_predicted = 0
    
    # Przypisanie każdej rozpoznanej gwiazdy do jej gwiazdozbioru
    for star in predicted_stars:
        constellation = sky_image.star_calculator.GetConstellation(star["predicted_star_id"])
        if constellation is not None:
            constellation_id = constellation["constellation_id"]
            star["constellation_id"] = constellation_id
            constellation["stars"].append(star)
            if constellation_id not in sky_image.constellations:
                sky_image.constellations[constellation_id] = constellation
            print(f"Gwiazda: {star['predicted_star_id']} Gwiazdozbiór: {constellation_id}")
        processed_predicted += 1
        if progress_callback is not None and total_predicted > 0:
            progress_callback("Łączenie gwiazd z gwiazdozbiorami", processed_predicted, total_predicted)

    # Obliczanie pozycji brakujących gwiazd w liniach gwiazdozbiorów
    total_lines = sum(len(const_data["lines"]) for const_data in sky_image.constellations.values())
    processed_lines = 0
    for const_id, const_data in sky_image.constellations.items():
        lines = const_data["lines"]
        for line in lines:
            # Dla każdej gwiazdy w linii gwiazdozbioru
            for star_id in line:
                found_star = sky_image.FindStarByHIP(star_id)
                # Jeśli gwiazda nie została znaleziona, oblicz jej pozycję geometrycznie
                if found_star is None:
                    CalculateStar(sky_image, star_id)
            processed_lines += 1
            if progress_callback is not None and total_lines > 0:
                progress_callback("Obliczanie brakujących gwiazd", processed_lines, total_lines)

    # Zbieranie statystyk rozpoznanych gwiazd według statusów
    sky_image.star_statistic = {}
    sky_image.star_statistic["Unknown"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["Unknown"]])
    sky_image.star_statistic["OK"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["OK"]])
    sky_image.star_statistic["Wrong"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["Wrong"]])
    sky_image.star_statistic["Recognized"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["Recognized"]])
    sky_image.star_statistic["Calculated"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["Calculated"]])
    sky_image.star_statistic["Assigned"] = len([s for s in sky_image.star_regions if s["status"] == sky_image.STAR_STATUS["Assigned"]])
    sky_image.star_statistic["Total"] = sum(sky_image.star_statistic.values())
    sky_image.is_recognized = True

def CalculateStar(sky_image, calculated_star_id: int):
    """
    Oblicza pozycję gwiazdy na obrazie na podstawie jej kątowych separacji od znanych gwiazd.
    Wykorzystuje triangulację z trzech najbliższych rozpoznanych gwiazd.
    """
    # Znajdowanie wszystkich rozpoznanych gwiazd i ich kątowych odległości od szukanej gwiazdy
    stars = {}
    for star in sky_image.star_regions:
        if star["status"] in [sky_image.STAR_STATUS["OK"], sky_image.STAR_STATUS["Wrong"], sky_image.STAR_STATUS["Recognized"]]:
            distance_factory = sky_image.star_calculator.GetAngularSeparation(star["predicted_star_id"], calculated_star_id)
            stars[star["predicted_star_id"]] = distance_factory

    # Potrzebujemy co najmniej 3 gwiazd do triangulacji
    if len(stars) < 3:
        return

    # Sortowanie gwiazd według odległości od szukanej gwiazdy
    stars = list(sorted(stars.items(), key=lambda item: item[1]))

    # Wybierz dwie najbliższe gwiazdy
    star_1 = sky_image.FindStarByHIP(stars[0][0])
    star_2 = sky_image.FindStarByHIP(stars[1][0])

    # Wybór trzeciej gwiazdy - ta, która tworzy najlepszy trójkąt (największy kąt)
    max_factory = 0.0
    best_star_3 = None
    for index in range(2, len(stars)):
        star_3 = sky_image.FindStarByHIP(stars[index][0])
        distance_1_3 = sky_image.DistanceBetweenStars(star_1, star_3)
        # Obliczenie kątowych separacji między gwiazdami (z katalogu)
        angular_1_2 = sky_image.star_calculator.GetAngularSeparation(star_1["predicted_star_id"], star_2["predicted_star_id"])
        angular_1_3 = sky_image.star_calculator.GetAngularSeparation(star_1["predicted_star_id"], star_3["predicted_star_id"])
        angular_2_3 = sky_image.star_calculator.GetAngularSeparation(star_2["predicted_star_id"], star_3["predicted_star_id"])
        # Obliczenie kątów trójkąta za pomocą twierdzenia cosinusów
        angle_1 = math.acos((angular_1_2 ** 2 + angular_1_3 ** 2 - angular_2_3 ** 2) / (2 * angular_1_2 * angular_1_3))
        angle_2 = math.acos((angular_1_2 ** 2 + angular_2_3 ** 2 - angular_1_3 ** 2) / (2 * angular_1_2 * angular_2_3))
        angle_3 = math.acos((angular_1_3 ** 2 + angular_2_3 ** 2 - angular_1_2 ** 2) / (2 * angular_1_3 * angular_2_3))
        # Normalizacja kątów do zakresu [0, π/2] (wybór mniejszego kąta)
        if angle_1 > math.pi / 2.0:
            angle_1 = math.pi - angle_1
        if angle_2 > math.pi / 2.0:
            angle_2 = math.pi - angle_2
        if angle_3 > math.pi / 2.0:
            angle_3 = math.pi - angle_3
        # Wybór największego kąta w trójkącie
        angle = max(angle_1, angle_2, angle_3)
        # Współczynnik jakości trójkąta (większy kąt i mniejsza odległość = lepiej)
        factory = angle / distance_1_3
        if factory > max_factory:
            max_factory = factory
            best_star_3 = star_3
    star_3 = best_star_3

    # Triangulacja: obliczenie pozycji szukanej gwiazdy
    distance_1_2 = sky_image.DistanceBetweenStars(star_1, star_2)
    distance_1_3 = sky_image.DistanceBetweenStars(star_1, star_3)
    # Kątowe odległości od szukanej gwiazdy do trzech gwiazd referencyjnych
    angular_1_x = sky_image.star_calculator.GetAngularSeparation(star_1["predicted_star_id"], calculated_star_id)
    angular_2_x = sky_image.star_calculator.GetAngularSeparation(star_2["predicted_star_id"], calculated_star_id)
    angular_3_x = sky_image.star_calculator.GetAngularSeparation(star_3["predicted_star_id"], calculated_star_id)
    angular_1_2 = sky_image.star_calculator.GetAngularSeparation(star_1["predicted_star_id"], star_2["predicted_star_id"])
    angular_1_3 = sky_image.star_calculator.GetAngularSeparation(star_1["predicted_star_id"], star_3["predicted_star_id"])
    # Przeliczenie kątowych odległości na odległości w pikselach na obrazie
    distance_1_x = (distance_1_2 * angular_1_x) / angular_1_2
    distance_2_x = (distance_1_2 * angular_2_x) / angular_1_2
    distance_3_x = (distance_1_3 * angular_3_x) / angular_1_3
    # Znalezienie punktów przecięcia okręgów wokół star_1 i star_2
    intersections_1_2 = FindCircleIntersections(star_1, star_2, distance_1_2, distance_1_x, distance_2_x)
    # Wybór punktu przecięcia najbliższego do oczekiwanej odległości od star_3
    min_diff = float("inf")
    selected_intersection = None
    for point_1_2 in intersections_1_2:
        # Obliczenie różnicy między rzeczywistą a oczekiwaną odległością od star_3
        diff = abs(math.hypot(point_1_2[0] - star_3["X"], point_1_2[1] - star_3["Y"]) - distance_3_x)
        if diff < min_diff:
            min_diff = diff
            selected_intersection = point_1_2

    if selected_intersection is None:
        return

    # Sprawdzenie, czy w obliczonej pozycji istnieje już nierozpoznana gwiazda
    existed_star = FindStarByPosition(
        sky_image,
        selected_intersection[0],
        selected_intersection[1],
        sky_image.max_error_distance * (distance_1_x + distance_2_x) / 2.0,
    )
    if existed_star is not None:
        # Jeśli znaleziono istniejącą gwiazdę, przypisz jej obliczone ID
        existed_star["star_id"] = calculated_star_id
        existed_star["calculated_star_id"] = calculated_star_id
        existed_star["constellation_id"] = sky_image.star_calculator.GetConstellation(calculated_star_id)["constellation_id"]
        existed_star["status"] = sky_image.STAR_STATUS["Assigned"]
    else:
        # Jeśli nie znaleziono, utwórz nowy wpis gwiazdy
        # Sprawdzenie, czy obliczona pozycja znajduje się w granicach obrazu
        if (
            selected_intersection[0] < 0
            or selected_intersection[0] >= sky_image.columns
            or selected_intersection[1] < 0
            or selected_intersection[1] >= sky_image.rows
        ):
            status = sky_image.STAR_STATUS["Outside"]
        else:
            status = sky_image.STAR_STATUS["Calculated"]

        star = {
            "star_id": calculated_star_id,
            "expected_star_id": 0,
            "predicted_star_id": 0,
            "calculated_star_id": calculated_star_id,
            "constellation_id": sky_image.star_calculator.GetConstellation(calculated_star_id)["constellation_id"],
            "minX": int(selected_intersection[0] - 1.0),
            "maxX": int(selected_intersection[0] + 1.0),
            "minY": int(selected_intersection[1] - 1.0),
            "maxY": int(selected_intersection[1] + 1.0),
            "X": selected_intersection[0],
            "Y": selected_intersection[1],
            "status": status,
        }
        sky_image.star_regions.append(star)

def FindCircleIntersections(star_1, star_2, distance_1_2, distance_1_x, distance_2_x):
    """
    Znajduje punkty przecięcia dwóch okręgów.
    Używane do triangulacji pozycji gwiazdy na podstawie odległości od dwóch znanych punktów.
    """
    circle_intersections = []

    # Korekta odległości, aby spełniały nierówność trójkąta (zapobieganie błędom numerycznym)
    if distance_1_2 + distance_1_x < distance_2_x * 1.0001:
        distance_2_x = (distance_1_2 + distance_1_x) / 1.0001
    if distance_1_2 + distance_2_x < distance_1_x * 1.0001:
        distance_1_x = (distance_1_2 + distance_2_x) / 1.0001
    if distance_1_x + distance_2_x < distance_1_2 * 1.0001:
        distance_1_2 = (distance_1_x + distance_2_x) / 1.0001

    # Obliczenia geometryczne przecięcia dwóch okręgów
    d = distance_1_2  # Odległość między środkami okręgów
    # Odległość od star_1 do punktu na linii między star_1 i star_2 prostopadłego do punktów przecięcia
    a = (distance_1_x ** 2 - distance_2_x ** 2 + d ** 2) / (2 * d)
    h_sq = distance_1_x ** 2 - a ** 2  # Wysokość od linii do punktów przecięcia (kwadrat)
    if h_sq >= 0:
        h = math.sqrt(max(h_sq, 0.0))
        x0, y0 = star_1["X"], star_1["Y"]
        x1, y1 = star_2["X"], star_2["Y"]
        # Punkt środkowy między punktami przecięcia (na linii star_1 - star_2)
        xm = x0 + a * (x1 - x0) / d
        ym = y0 + a * (y1 - y0) / d
        # Wektor prostopadły do linii star_1 - star_2
        rx = -(y1 - y0) * (h / d)
        ry = (x1 - x0) * (h / d)
        # Dwa punkty przecięcia (lub jeden, jeśli h = 0)
        circle_intersections.append((xm + rx, ym + ry))
        if h > 0:
            circle_intersections.append((xm - rx, ym - ry))

    return circle_intersections

def FindStarByPosition(sky_image, x: float, y: float, max_distance: float):
    """
    Znajduje nierozpoznaną gwiazdę w zadanej pozycji (w określonym promieniu).
    Używane do przypisania obliczonego ID do istniejącej, ale nierozpoznanej gwiazdy.
    """
    for star in sky_image.star_regions:
        # Szukamy tylko nierozpoznanych lub pominiętych gwiazd
        if star["status"] in [sky_image.STAR_STATUS["Unknown"], sky_image.STAR_STATUS["Missed"]]:
            distance = sky_image.DistanceBetweenStars(star, {"X": x, "Y": y})
            if distance <= max_distance:
                return star
    return None
