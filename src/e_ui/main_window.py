from importlib.resources import files
import os
import joblib
from pathlib import Path

from b_image_processing.StarCalculation import StarCalculator
from b_image_processing.SkyImage import SkyImage
import c_stars_recognizing.RecognizeStars as rs
import d_constellations_recognizing.RecognizeConstellations as rc

from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QMenuBar,
    QMenu,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QSizePolicy,
)
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QPainter, QPen, QColor, QShortcut


# Zmienna kontrolująca sposób wyświetlania rozpoznanych gwiazd w podsumowaniu
# 0 = bez podziału (łącznie jako "rozpoznanych")
# 1 = z podziałem (osobno "poprawnie rozpoznanych" i "błędnie rozpoznanych"), dotyczy tylko zdjęć syntetycznych i pozwala zobaczyć poprawność rozpoznania z poziomu interfejsu 
SHOW_DETAILED_RECOGNITION = 0

_STATUS_COLOR_SPECS = [
    ("Unknown", Qt.white),
    ("OK", Qt.green),
    ("Wrong", Qt.magenta),
    ("Recognized", Qt.green),
    ("Calculated", Qt.red),
    ("Assigned", Qt.darkYellow),
    ("Missed", Qt.white),
    ("Verified", Qt.cyan),
]

STAR_STATUS_COLORS = {
    SkyImage.STAR_STATUS[name]: color
    for name, color in _STATUS_COLOR_SPECS
    if name in SkyImage.STAR_STATUS
}


class RecognitionLegend(QWidget):
    """Stały pasek legendy pokazujący marker gwiazd rozpoznanych."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: #000000;")
        self._items: list[tuple[QColor, str]] = []
        self.set_items([])

    def set_items(self, items):
        """Ustawia listę markerów: elementy mogą zawierać kolor lub status."""
        normalized: list[tuple[QColor, str]] = []
        for item in items:
            color = None
            label = ""

            if isinstance(item, dict):
                label = item.get("label", "")
                color_value = item.get("color")
                status_value = item.get("status")
            elif isinstance(item, tuple) and len(item) >= 2:
                color_value = item[0]
                label = item[1]
                status_value = None
            else:
                continue

            if color_value is None and status_value is not None:
                color_value = STAR_STATUS_COLORS.get(status_value)

            if color_value is None:
                continue

            color = QColor(color_value)
            normalized.append((color, label))

        self._items = normalized
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(5, 8, 20))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        text_pen = QPen(Qt.white)

        cy = self.height() // 2
        metrics = painter.fontMetrics()
        spacing_element = 24
        spacing_text = 16
        x = 20

        for color, label in self._items:
            self._draw_marker(painter, x, cy, color)
            x += spacing_text
            painter.setPen(text_pen)
            text_y = (self.height() + metrics.ascent() - metrics.descent()) // 2
            painter.drawText(x, text_y, label)
            x += metrics.horizontalAdvance(label) + spacing_element

        painter.end()

    def _draw_marker(self, painter: QPainter, cx: int, cy: int, color: QColor):
        offset = 3
        length = 4
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(cx + offset, cy + offset, cx + offset + length, cy + offset + length)
        painter.drawLine(cx - offset, cy - offset, cx - offset - length, cy - offset - length)
        painter.drawLine(cx + offset, cy - offset, cx + offset + length, cy - offset - length)
        painter.drawLine(cx - offset, cy + offset, cx - offset - length, cy + offset + length)        


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikacja do rozpoznawania gwiazdozbiorów na zdjęciach nieba")
        self.setGeometry(100, 100, 1000, 700)
        self.sky_image = None
        self.original_pixmap = None
        self.star_calculator = StarCalculator()
        self.show_star_markers = True
        self.show_star_labels = True
        self.show_constellations = True
        self._processing_frames = ["|", "/", "-", "\\"]
        self._processing_index = 0
        self._processing_message = ""
        self.processing_label = None
        self.processing_timer = None
        self.setup_ui()
        self.create_menu_bar()
        self.setup_shortcuts()
        self.model_path = "./model_final_knn.pkl"
        self.model_path = Path(__file__).resolve().parent / ".." / "c_stars_recognizing" / "model_final_knn.pkl"
        # if os.path.isfile(self.model_path):
        self.model =  joblib.load(self.model_path)
        self.isRecognized = False
        self.image_scale = 1.0
        self.file_name = ""
        self.folder = ""
        self.files = []
        self.current_file_index = -1

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        self.splitter = QSplitter(Qt.Horizontal)

        self.image_label = QLabel("Wczytaj obraz nieba, aby rozpocząć")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #eef2f5; border: 1px solid #bdc3c7; font-size: 16px; color: #2c3e50;"
        )
        self.image_label.setMinimumSize(400, 400)

        self.recognized_legend = RecognitionLegend()
        self.recognized_legend.set_items([])

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.recognized_legend)
        left_layout.setStretch(0, 1)

        self.info_panel = QLabel("")
        self.info_panel.setAttribute(Qt.WA_StyledBackground, True)
        self.info_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_panel.setStyleSheet(
            "padding: 12px; font-size: 14px; color: #000000; background-color: #ffffff; border: 1px solid #bdc3c7;"
        )
        self.info_panel.setWordWrap(True)

        self.discovery_label = QLabel("Model może mylić się.")
        self.discovery_label.setAlignment(Qt.AlignCenter)
        self.discovery_label.setStyleSheet(
            "padding: 8px; font-size: 12px; color: #2c3e50; background-color: #f7f9fb; border: 1px solid #d1d9e6;"
        )
        self.discovery_label.setWordWrap(True)

        self.processing_label = QLabel("")
        self.processing_label.setAlignment(Qt.AlignCenter)
        self.processing_label.setStyleSheet(
            "padding: 8px; font-size: 12px; color: #1d3557; background-color: #ffb3b3; border: 1px solid #ff5c5c;"
        )
        self.processing_label.setWordWrap(True)
        self.processing_label.hide()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.info_panel)
        right_layout.addWidget(self.discovery_label)
        right_layout.addWidget(self.processing_label)
        right_layout.setStretch(0, 1)

        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([800, 200])

        layout.addWidget(self.splitter)

        central_widget.setLayout(layout)
        self.processing_timer = QTimer(self)
        self.processing_timer.setInterval(150)
        self.processing_timer.timeout.connect(self._on_processing_timer)
    
    def create_menu_bar(self):
        """Tworzy pasek menu z opcjami"""
        menubar = self.menuBar()
        
        # Menu Plik
        file_menu = menubar.addMenu("&Plik")

        # Akcja Otwórz
        open_file_action = QAction("&Otwórz obraz", self)
        open_file_action.setShortcut(QKeySequence.Open)
        open_file_action.setStatusTip("Otwórz plik obrazu do rozpoznania")
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)

        file_menu.addSeparator()

        first_file_action = QAction("&Pierwszy obraz", self)
        first_file_action.setShortcut("Ctrl+Home")
        first_file_action.setStatusTip("Przejdź do pierwszego obrazu w folderze")
        first_file_action.triggered.connect(self.show_first_image)
        file_menu.addAction(first_file_action)

        previous_file_action = QAction("&Poprzedni obraz", self)
        previous_file_action.setShortcut("Ctrl+Left")
        previous_file_action.setStatusTip("Przejdź do poprzedniego obrazu w folderze")
        previous_file_action.triggered.connect(self.show_previous_image)
        file_menu.addAction(previous_file_action)

        next_file_action = QAction("&Następny obraz", self)
        next_file_action.setShortcut("Ctrl+Right")
        next_file_action.setStatusTip("Przejdź do następnego obrazu w folderze")
        next_file_action.triggered.connect(self.show_next_image)
        file_menu.addAction(next_file_action)

        last_file_action = QAction("&Ostatni obraz", self)
        last_file_action.setShortcut("Ctrl+End")
        last_file_action.setStatusTip("Przejdź do ostatniego obrazu w folderze")
        last_file_action.triggered.connect(self.show_last_image)
        file_menu.addAction(last_file_action)

        file_menu.addSeparator()

        # Akcja Otwórz folder
        open_folder_action = QAction("Otwórz &folder", self)
        open_folder_action.setShortcut("Ctrl+F")
        open_folder_action.setStatusTip("Otwórz folder obrazów do rozpoznania")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        # Separator
        file_menu.addSeparator()
        
        # Akcja Zakończ
        exit_action = QAction("&Zakończ", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Zakończ działanie aplikacji")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Rozpoznawanie
        recognition_menu = menubar.addMenu("&Rozpoznawanie")
        
        # Akcja Rozpoznaj gwiazdy
        recognize_action = QAction("&Rozpoznaj gwiazdozbiory", self)
        recognize_action.setShortcut("Ctrl+R")
        recognize_action.setStatusTip("Rozpoznaj gwiazdozbiory na obrazie")
        recognize_action.triggered.connect(self.recognize_constellations)
        recognition_menu.addAction(recognize_action)

        # Menu Widok
        view_menu = menubar.addMenu("&Widok")

        show_markers_action = QAction("Pokaż znaczniki gwiazd", self, checkable=True)
        show_markers_action.setShortcut("Ctrl+1")
        show_markers_action.setStatusTip("Pokaż lub ukryj znaczniki gwiazd na obrazie")
        show_markers_action.setChecked(self.show_star_markers)
        show_markers_action.triggered.connect(self.toggle_star_markers)
        view_menu.addAction(show_markers_action)

        show_labels_action = QAction("Pokaż nazwy gwiazd", self, checkable=True)
        show_labels_action.setShortcut("Ctrl+2")
        show_labels_action.setStatusTip("Pokaż lub ukryj nazwy gwiazd na obrazie")
        show_labels_action.setChecked(self.show_star_labels)
        show_labels_action.triggered.connect(self.toggle_star_labels)
        view_menu.addAction(show_labels_action)

        show_constellations_action = QAction("Pokaż linie i nazwy gwiazdozbiorów", self, checkable=True)
        show_constellations_action.setShortcut("Ctrl+3")
        show_constellations_action.setStatusTip("Pokaż lub ukryj linie i nazwy gwiazdozbiorów na obrazie")
        show_constellations_action.setChecked(self.show_constellations)
        show_constellations_action.triggered.connect(self.toggle_constellations)
        view_menu.addAction(show_constellations_action)

        # Menu Pomoc
        help_menu = menubar.addMenu("&Pomoc")
        
        # Akcja O programie
        about_action = QAction("&O programie", self)
        about_action.setStatusTip("Pokaż informacje o aplikacji")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Dodanie paska statusu
        status_bar = self.statusBar()
        font = status_bar.font()
        font.setPointSize(int(font.pointSize() * 1.3) if font.pointSize() > 0 else 14)
        status_bar.setFont(font)
        status_bar.showMessage("Nie wykonano żadnej akcji")
    
    def setup_shortcuts(self):
        self.stop_recognition_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_recognition_shortcut.activated.connect(self.stop_recognition)

  
    def open_file(self):
        """Otwiera obraz PNG, wyświetla go oraz tworzy obiekt SkyImage2."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz obraz nieba",
            "",
            "Obrazy (*.png *.jpg *.jpeg *.bmp)",
        )

        if not file_path:
            return

        try:

            self._load_selected_image(file_path)
            self.files = [file_path]
            self.current_file_index = 0
        except Exception as exc:
            QMessageBox.critical(self, "Błąd wczytywania", str(exc))
            self.statusBar().showMessage("Błąd wczytywania obrazu")

    def _load_selected_image(self, file_path):
        self.folder = os.path.dirname(file_path)
        self.file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(self.file_name)[1]

        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            self.image_label.clear()
        self.image_label.setText("Ładowanie i analizowanie obrazu...")
        self.info_panel.setText("\n".join(
                [
                    f"Plik: {self.file_name}",
                    f"Folder: {self.folder}",
                ]
            )
        )       
        self.statusBar().showMessage("Ładowanie obrazu...")
        QApplication.processEvents()
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            raise ValueError("Nie udało się wczytać obrazu.")

        ## odczytanie zdjęcia
        self.original_pixmap = pixmap
        self.isRecognized = False


        # utworzenie obiektu SkyImage2
        self.statusBar().showMessage("Analiza obrazu...")
        QApplication.processEvents()
        self.sky_image = SkyImage(self.folder, self.file_name, file_extension)
        
        ## wyświetlenie obrazu
        self._update_image_display()

        ## aktualizacja panelu informacji
        self.statusBar().showMessage(f"Załadowano obraz: {self.file_name}")

        text = "\n".join(
                [
                    f"Plik: {self.file_name}",
                    f"Folder: {self.folder}",
                    f"Rozmiar: {pixmap.width()}x{pixmap.height()} px",
                    f"\nGwiazd:",
                    f"{len(self.sky_image.star_regions)} - wykrytych",
                ]
            )
        
        if len(self.sky_image.star_regions) > 30:
            text += "\n\nWykryto dużo gwiazd, dlatego rozpoznawanie może potrwać znacznie dłużej."

        self.info_panel.setText(text)

        self.recognized_legend.set_items([
            {"status": SkyImage.STAR_STATUS.get("Unknown"), "label": "Gwiazda nierozpoznana"},
        ])           
    
    def open_folder(self):
        """Otwiera folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami", "")
        if not folder_path:
            return

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [
            os.path.join(folder_path, name)
            for name in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, name))
            and os.path.splitext(name)[1].lower() in valid_ext
        ]

        if not files:
            QMessageBox.information(self, "Brak plików", "Folder nie zawiera obsługiwanych obrazów.")
            return

        self.files = files
        self.current_file_index = 0
        self._load_selected_image(self.files[0])

    def show_next_image(self):
        if not self.files:
            self.statusBar().showMessage("Brak kolejnych obrazów")
            return

        if self.current_file_index < len(self.files) - 1:
            self.current_file_index += 1
            self._load_selected_image(self.files[self.current_file_index])
            self.statusBar().showMessage(f"Załadowano obraz: {self.file_name}")
        else:
            self.statusBar().showMessage("To ostatni obraz w folderze")

    def show_previous_image(self):
        if not self.files:
            self.statusBar().showMessage("Brak poprzednich obrazów")
            return

        if self.current_file_index > 0:
            self.current_file_index -= 1
            self._load_selected_image(self.files[self.current_file_index])
            self.statusBar().showMessage(f"Załadowano obraz: {self.file_name}")
        else:
            self.statusBar().showMessage("To pierwszy obraz w folderze")

    def show_first_image(self):
        if not self.files:
            self.statusBar().showMessage("Brak obrazów do wyświetlenia")
            return

        if self.current_file_index != 0:
            self.current_file_index = 0
            self._load_selected_image(self.files[self.current_file_index])
        self.statusBar().showMessage(f"Załadowano obraz: {self.file_name}")

    def show_last_image(self):
        if not self.files:
            self.statusBar().showMessage("Brak obrazów do wyświetlenia")
            return

        last_index = len(self.files) - 1
        if self.current_file_index != last_index:
            self.current_file_index = last_index
            self._load_selected_image(self.files[self.current_file_index])
        self.statusBar().showMessage(f"Załadowano obraz: {self.file_name}")

    def toggle_star_markers(self, checked: bool):
        self.show_star_markers = checked
        self._update_image_display()

    def toggle_star_labels(self, checked: bool):
        self.show_star_labels = checked
        self._update_image_display()

    def toggle_constellations(self, checked: bool):
        self.show_constellations = checked
        self._update_image_display()

    def start_processing_animation(self, message: str):
        self._processing_index = 0
        self._processing_message = message
        if self.processing_label is None:
            return
        self.processing_label.setText(f"{self._processing_frames[self._processing_index]} {self._processing_message}")
        self.processing_label.show()
        if self.processing_timer is not None and not self.processing_timer.isActive():
            self.processing_timer.start()

    def update_processing_animation(self, message: str):
        self._processing_message = message
        if self.processing_label is not None and self.processing_label.isVisible():
            self.processing_label.setText(f"{self._processing_frames[self._processing_index]} {self._processing_message}")

    def stop_processing_animation(self):
        if self.processing_timer is not None and self.processing_timer.isActive():
            self.processing_timer.stop()
        if self.processing_label is not None:
            self.processing_label.clear()
            self.processing_label.hide()
        self._processing_message = ""

    def _on_processing_timer(self):
        if self.processing_label is None or not self.processing_label.isVisible():
            return
        self._processing_index = (self._processing_index + 1) % len(self._processing_frames)
        self.processing_label.setText(f"{self._processing_frames[self._processing_index]} {self._processing_message}")

    def _on_recognition_progress(self, phase: str, current: int, total: int):
        if total > 0:
            percent = int((current / total) * 100)
            message = f"{phase} {current}/{total} ({percent}%)"
        else:
            message = phase
        self.update_processing_animation(message)
        QApplication.processEvents()

    def stop_recognition(self):
        if self.sky_image is None:
            self.statusBar().showMessage("Brak aktywnego rozpoznawania")
            return

        self.sky_image.break_recognition = True
        self.stop_processing_animation()
        self.statusBar().showMessage("Przerwano rozpoznawanie gwiazd")
    
    def recognize_constellations(self):
        """Rozpoznaje gwiazdozbiory"""
        if self.sky_image is None:
            QMessageBox.warning(self, "Rozpoznawanie gwiazd", "Najpierw wczytaj obraz PNG.")
            return

        if self.sky_image.star_regions is None or len(self.sky_image.star_regions) < 4:
            QMessageBox.warning(self, "Rozpoznawanie gwiazd", "Zbyt mało gwiazd do rozpoznania gwiazdozbiorów.")
            return

        if self.sky_image.is_recognized:
            QMessageBox.warning(self, "Rozpoznawanie gwiazd", "Gwiazdozbiory zostały już rozpoznane.")
            return
        
        self.start_processing_animation("Rozpoznawanie gwiazd...")
        try:
            self.statusBar().showMessage("Rozpoznawanie gwiazd...")
            QApplication.processEvents()
            recognized = rs.RecognizeStars(self.sky_image,self.model, self._on_recognition_progress)
            if not recognized:
                self.statusBar().showMessage("Rozpoznawanie przerwane przez użytkownika")
                return

            self.update_processing_animation("Rozpoznawanie gwiazdozbiorów...")
            self.statusBar().showMessage("Rozpoznawanie gwiazdozbiorów...")
            QApplication.processEvents()
            rc.RecognizeConstellations(self.sky_image, self.model, self._on_recognition_progress)
            self.isRecognized = True
        finally:
            self.stop_processing_animation()

        self._update_image_display()
        stars_found = len(getattr(self.sky_image, "star_regions", []))
        text = [
                    f"Plik: {self.file_name}",
                    f"Folder: {self.folder}",
                    f"Rozmiar: {self.sky_image.source_image.width}x{self.sky_image.source_image.height} px",
                    f"\nGwiazd:",
        ]

        legend = []

        # Sprawdzenie czy pokazywać podział na OK/Wrong czy łącznie jako Recognized
        if SHOW_DETAILED_RECOGNITION == 1:
            # Z podziałem na poprawne i błędne
            if self.sky_image.star_statistic['OK'] > 0:
                text.append(f"{self.sky_image.star_statistic['OK']} - poprawnie rozpoznanych")
                legend.append({"status": SkyImage.STAR_STATUS.get("OK"), "label": "Gwiazda poprawnie rozpoznana"})
            if self.sky_image.star_statistic['Wrong'] > 0:
                text.append(f"{self.sky_image.star_statistic['Wrong']} - błędnie rozpoznanych")
                legend.append({"status": SkyImage.STAR_STATUS.get("Wrong"), "label": "Gwiazda błędnie rozpoznana"})
        else:
            # Bez podziału - łącznie jako Recognized
            total_recognized = self.sky_image.star_statistic.get('OK', 0) + self.sky_image.star_statistic.get('Wrong', 0) + self.sky_image.star_statistic.get('Recognized', 0)
            if total_recognized > 0:
                text.append(f"{total_recognized} - rozpoznanych")
                legend.append({"status": SkyImage.STAR_STATUS.get("Recognized"), "label": "Gwiazda rozpoznana"})
        
        if self.sky_image.star_statistic['Calculated'] > 0:
            text.append(f"{self.sky_image.star_statistic['Calculated']} - obliczonych")
            legend.append({"status": SkyImage.STAR_STATUS.get("Calculated"), "label": "Gwiazda obliczona"})
        if self.sky_image.star_statistic['Assigned'] > 0:
            text.append(f"{self.sky_image.star_statistic['Assigned']} - przypisanych")
            legend.append({"status": SkyImage.STAR_STATUS.get("Assigned"), "label": "Gwiazda przypisana"})
        if self.sky_image.star_statistic['Unknown'] > 0:
            text.append(f"{self.sky_image.star_statistic['Unknown']} - nierozpoznanych")
            legend.append({"status": SkyImage.STAR_STATUS.get("Unknown"), "label": "Gwiazda nierozpoznana"})
            
        text.append(f"{self.sky_image.star_statistic['Total']} - WSZYSTKICH")
        self.info_panel.setText("\n".join(text))

        constellations = sorted([x['constellation_name'] for x in self.sky_image.constellations.values()])
        if constellations:
            self.info_panel.setText(
                self.info_panel.text()
                + "\n\nRozpoznane gwiazdozbiory:\n- "
                + "\n- ".join(constellations)
            )

        self.recognized_legend.set_items(legend)
        self.statusBar().showMessage("Rozpoznawanie zakończone")
    
    def resizeEvent(self, event):  
        super().resizeEvent(event)
        self._update_image_display()

    def _update_image_display(self):
        """Aktualizuje widok obrazu, zachowując proporcje."""
        if self.original_pixmap is None:
            return

        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.image_scale = min(
            target_size.width() / self.original_pixmap.width(),
            target_size.height() / self.original_pixmap.height(),
        )
        # print(f"Image scale set to: {self.image_scale}")

        source = self.original_pixmap
        if self.show_star_markers:
            source = self._add_star_markers(source)
        if self.isRecognized:
            if self.show_star_labels:
                source = self._add_star_labels(source)
            if self.show_constellations:
                source = self._add_constellation(source)
        if source is None:
            source = self.original_pixmap

        scaled = source.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _get_color(self,star_region: dict):
        default_status = SkyImage.STAR_STATUS.get("Unknown")
        status = star_region.get("status", default_status)
        return STAR_STATUS_COLORS.get(status, Qt.white)

    def _add_star_markers(self, source_pixmap) -> QPixmap | None:
        """Zwraca kopię obrazu z narysowanymi markerami gwiazd."""
        offset = int(5 / self.image_scale)
        length = int(5 / self.image_scale)
        pen_width = 2.0 / self.image_scale
        target_pixmap = source_pixmap.copy()

        if self.sky_image is None or not getattr(self.sky_image, "star_regions", None):
            return target_pixmap

        painter = QPainter(target_pixmap)
        pen = QPen()
        pen.setWidth(pen_width)

        status_fallback = SkyImage.STAR_STATUS.get("Unknown", 0)
        stars = sorted(
            self.sky_image.star_regions,
            key=lambda region: region.get("status", SkyImage.STAR_STATUS["Unknown"]),
        )
        for star_region in stars:
            if star_region['status'] != SkyImage.STAR_STATUS['Outside']:
                pen.setColor(self._get_color(star_region))
                painter.setPen(pen)
                cx = int(round(star_region.get("X")))
                cy = int(round(star_region.get("Y")))
                painter.drawLine(cx + offset, cy + offset, cx + offset + length, cy + offset + length)
                painter.drawLine(cx - offset, cy - offset, cx - offset - length, cy - offset - length)
                painter.drawLine(cx + offset, cy - offset, cx + offset + length, cy - offset - length)
                painter.drawLine(cx - offset, cy + offset, cx - offset - length, cy + offset + length)

        painter.end()
        return target_pixmap


    def _add_star_labels(self, source_pixmap) -> QPixmap | None:
        """Zwraca kopię obrazu z dodanymi nazwami gwiazd."""
        offset = int(5 / self.image_scale)
        font_size = 12 / self.image_scale
        target_pixmap = source_pixmap.copy()
        if self.sky_image is None or not getattr(self.sky_image, "star_regions", None):
            return target_pixmap

        painter = QPainter(target_pixmap)
        painter.setPen(QPen(Qt.white))
        font = painter.font()
        font.setPixelSize(font_size)
        painter.setFont(font)

        for star_region in self.sky_image.star_regions:

            star_id = int(star_region.get("star_id") or 0)
            expected_id = int(star_region.get("expected_star_id") or 0)
            if star_id > 0:
                cx = int(round(star_region.get("X")))
                cy = int(round(star_region.get("Y")))
                # if expected_id > 0 and star_id != expected_id:
                #     label = f"{self.star_calculator.get_star_name(star_id)} ({star_id} -> {star_region.get("expected_star_id")})"
                # else:
                label = f"{self.star_calculator.GetStarName(star_id)} ({star_id})"
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label)
                text_height = metrics.height()
                # print(f"Text '{label}' width: {text_width}, height: {text_height}")
                x_text = int(cx - text_width / 2)
                y_text = int(cy + offset + text_height)
                if x_text >= 0 and y_text >= 0 and x_text + text_width <= target_pixmap.width() and y_text - text_height >= 0:
                    painter.drawText(x_text, y_text, label)
                # painter.restore()


        painter.end()
        return target_pixmap
    
    def _add_constellation(self, source_pixmap) -> QPixmap | None:
        target_pixmap = source_pixmap.copy()
        painter = QPainter(target_pixmap)
        pen = QPen(Qt.cyan)
        pen.setWidth(1.0 / self.image_scale)
        painter.setPen(pen)
        stars = self.sky_image.star_regions

        for cid, constellation in self.sky_image.constellations.items():
            lines = constellation.get('lines', [])

            # Rysowanie linii gwiazdozbiorów
            for line in lines:
                start_star_id, end_star_id = line
                start_star = self.sky_image.FindStarByHIP(start_star_id)
                end_star = self.sky_image.FindStarByHIP(end_star_id)
                if start_star is None or end_star is None:
                    continue
                cx1 = int(round(start_star.get("X")))
                cy1 = int(round(start_star.get("Y")))
                cx2 = int(round(end_star.get("X")))
                cy2 = int(round(end_star.get("Y")))
                painter.drawLine(cx1, cy1, cx2, cy2)
            
            # Rysowanie nazwy gwiazdozbioru
            minX = min([s['X'] for s in stars if s.get('constellation_id','') == cid and s['status'] != SkyImage.STAR_STATUS['Outside']])
            maxX = max([s['X'] for s in stars if s.get('constellation_id','') == cid and s['status'] != SkyImage.STAR_STATUS['Outside']])
            minY = min([s['Y'] for s in stars if s.get('constellation_id','') == cid and s['status'] != SkyImage.STAR_STATUS['Outside']])
            maxY = max([s['Y'] for s in stars if s.get('constellation_id','') == cid and s['status'] != SkyImage.STAR_STATUS['Outside']])
            cx = int(round((minX + maxX) / 2))
            cy = int(round((minY + maxY) / 2))
            # if constellation.get('stars'):

            #     posX = [s['X'] for s in constellation.get('stars', []) if s]
            #     posY = [s['Y'] for s in constellation.get('stars', []) if s]
            #     cx = int(round(sum(posX) / len(posX))) if posX else 0
            #     cy = int(round(sum(posY) / len(posY))) if posY else 0

            font_size = 20 / self.image_scale
            font = painter.font()
            font.setPixelSize(font_size)
            painter.setFont(font)
            label = constellation.get('constellation_name', 'Unknown')
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label)
            text_height = metrics.height()
            x_text = int(max(0, min(cx - text_width / 2, target_pixmap.width() - text_width)))
            y_text = int(max(text_height, min(cy - 5, target_pixmap.height() - 2)))
            painter.drawText(x_text, y_text, label)

        painter.end()        
        return target_pixmap

    
    
    def show_about(self):
        """Pokazuje informacje o programie"""
        QMessageBox.about(self, "O programie", 
                         "Gwiazdozbiory v1.0\n\n"
                         "Aplikacja do rozpoznawania gwiazdozbiorów\n"
                         "z użyciem uczenia maszynowego.\n\n"
                         "Autor: Weronika Jankowiak\n"
                         "Data: 2025-2026")