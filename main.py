import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QColorDialog, QFrame, QSizePolicy, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QCursor, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QSize

from image_processing import detect_base_color, convert_negative, apply_adjustments

import rawpy


class HistogramCanvas(FigureCanvas):
    """Canvas for displaying image histograms using matplotlib."""

    def __init__(self, parent=None, width=3, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        self.fig.patch.set_facecolor('white')
        self.axes.set_facecolor('white')
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    def update_histogram(self, img, mode='rgb'):
        """Update the histogram plot for the given image.

        Args:
            img (ndarray): Input image.
            mode (str): 'rgb' or 'luminance' mode.
        """
        self.axes.clear()
        if img is not None:
            if mode == 'rgb':
                for i, color in enumerate(('r', 'g', 'b')):
                    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                    self.axes.plot(hist, color=color)
                self.axes.set_title('RGB', fontsize=8)
            elif mode == 'luminance':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                self.axes.plot(hist, color='black')
                self.axes.set_title('Luminance', fontsize=8)
            self.axes.set_xlim([0, 256])
            self.axes.grid(False)
            self.axes.set_xlabel('Pixel Value', fontsize=7)
            self.axes.set_ylabel('Frequency', fontsize=7)
            self.axes.tick_params(labelsize=6)
        self.draw()


class NegativeConverter(QMainWindow):
    """Main application window for the NEGATRON image converter."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEGATRON v1.0")
        self.setWindowFlags(Qt.FramelessWindowHint)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(script_dir, 'fonts', 'Bowman.ttf')
        nega_font_path = os.path.join(script_dir, 'fonts', 'nega.ttf')

        font_id = QFontDatabase.addApplicationFont(font_path)
        self.digital_font_family = (QFontDatabase.applicationFontFamilies(font_id)[0]
                                      if font_id != -1 else QFont().family())
        nega_font_id = QFontDatabase.addApplicationFont(nega_font_path)
        self.nega_font_family = (QFontDatabase.applicationFontFamilies(nega_font_id)[0]
                                 if nega_font_id != -1 else self.digital_font_family)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: white;
                color: black;
                font-family: '{self.digital_font_family}';
            }}
            QLabel {{
                background: transparent;
                font-family: '{self.digital_font_family}';
            }}
            QPushButton {{
                font-family: '{self.digital_font_family}';
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #000000;
                height: 4px;
                background: #000000;
                margin: 2px 0;
            }}
            QSlider::handle:horizontal {{
                background: #000000;
                border: 1px solid #000000;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #FFFFFF;
                border: 1px solid #000000;
            }}
            QSlider {{
                background: transparent;
            }}
        """)

        self.presets = {
            "Auto": None,
            "Kodak": (255, 128, 0),
            "Fuji": (150, 180, 100),
            "Ilford": (200, 200, 200)
        }

        self.image = None
        self.processed_full_resolution = None
        self.base_color = None
        self.auto_detected_color = None
        self.current_preset = "Auto"
        self.slider_update_in_progress = False

        self.hue_value = 0
        self.saturation_value = 100
        self.contrast_value = 100
        self.brightness_value = 100
        self.shadows_value = 100
        self.highlights_value = 100

        self.histogram_canvas = None
        self.luminance_histogram = None

        self.init_ui()

    def init_ui(self):
        """Initialize and arrange UI components."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel("NEGATRON v1")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont(self.nega_font_family)
        title_font.setPointSize(32)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: black; background: transparent; "
                                  f"font-family: '{self.nega_font_family}';")
        layout.addWidget(title_label)

        header_layout = QHBoxLayout()
        header_layout.addStretch()
        close_btn = QPushButton("esc")
        close_btn.setFixedSize(40, 40)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: black;
                border: 1px solid black;
                font-size: 16px;
                font-family: '{self.digital_font_family}';
            }}
            QPushButton:hover {{
                background-color: black;
                color: white;
            }}
        """)
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        header_layout.addWidget(close_btn)
        layout.addLayout(header_layout)

        # Image display area (Negative and Positive)
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: transparent;")
        image_layout = QHBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(20)

        negative_container = QWidget()
        negative_container.setStyleSheet("background-color: transparent;")
        negative_layout = QVBoxLayout(negative_container)
        negative_layout.setContentsMargins(0, 0, 0, 0)
        negative_title = QLabel("Negative")
        negative_title.setAlignment(Qt.AlignCenter)
        negative_title.setStyleSheet(f"color: black; font-size: 30px; font-weight: bold; "
                                     f"font-family: '{self.digital_font_family}'; "
                                     "background: transparent;")
        negative_layout.addWidget(negative_title)
        self.negative_label = QLabel()
        self.negative_label.setAlignment(Qt.AlignCenter)
        self.negative_label.setMinimumSize(300, 300)
        self.negative_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.negative_label.setStyleSheet("background: transparent;")
        negative_layout.addWidget(self.negative_label)

        positive_container = QWidget()
        positive_container.setStyleSheet("background-color: transparent;")
        positive_layout = QVBoxLayout(positive_container)
        positive_layout.setContentsMargins(0, 0, 0, 0)
        positive_title = QLabel("Positive")
        positive_title.setAlignment(Qt.AlignCenter)
        positive_title.setStyleSheet(f"color: black; font-size: 30px; font-weight: bold; "
                                     f"font-family: '{self.digital_font_family}'; "
                                     "background: transparent;")
        positive_layout.addWidget(positive_title)
        self.positive_label = QLabel()
        self.positive_label.setAlignment(Qt.AlignCenter)
        self.positive_label.setMinimumSize(300, 300)
        self.positive_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.positive_label.setStyleSheet("background: transparent;")
        positive_layout.addWidget(self.positive_label)

        image_layout.addWidget(negative_container)
        image_layout.addWidget(positive_container)
        layout.addWidget(image_frame, stretch=1)

        # Main control buttons
        main_controls = QHBoxLayout()
        main_controls.setSpacing(15)
        load_btn = QPushButton("Load")
        load_btn.setFixedSize(80, 40)
        load_btn.clicked.connect(self.load_image)
        self.style_button(load_btn)
        main_controls.addWidget(load_btn)
        main_controls.addStretch(1)
        convert_btn = QPushButton("Convert")
        convert_btn.setFixedSize(100, 50)
        convert_btn.clicked.connect(self.process_image)
        convert_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: black;
                border: 1px solid black;
                padding: 8px 15px;
                font-size: 16px;
                font-family: '{self.digital_font_family}';
            }}
            QPushButton:hover {{
                background-color: black;
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #333333;
                color: white;
            }}
        """)
        convert_btn.setCursor(QCursor(Qt.PointingHandCursor))
        main_controls.addWidget(convert_btn)
        main_controls.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.setFixedSize(80, 40)
        save_btn.clicked.connect(self.save_image)
        self.style_button(save_btn)
        main_controls.addWidget(save_btn)
        layout.addLayout(main_controls)

        # Sliders and histograms section
        sliders_container = QHBoxLayout()
        sliders_container.addStretch(1)
        slider_columns = QHBoxLayout()
        slider_columns.setSpacing(20)

        # Histograms (Luminance with levels adjustment and RGB)
        histograms_container = QHBoxLayout()
        histograms_container.setSpacing(15)
        luminance_container = QVBoxLayout()
        luminance_container.setSpacing(5)
        self.luminance_histogram = HistogramCanvas(self, width=2.5, height=2, dpi=80)
        self.luminance_histogram.setFixedSize(150, 120)
        luminance_container.addWidget(self.luminance_histogram)

        # Levels adjustment sliders
        levels_layout = QHBoxLayout()
        levels_layout.setSpacing(5)
        black_label = QLabel("Black:")
        black_label.setStyleSheet("font-size: 8pt; color: black; background: transparent;")
        levels_layout.addWidget(black_label)
        self.black_point_slider = QSlider(Qt.Horizontal)
        self.black_point_slider.setRange(0, 255)
        self.black_point_slider.setValue(0)
        self.black_point_slider.setFixedWidth(70)
        self.black_point_slider.valueChanged.connect(self.levels_slider_changed)
        levels_layout.addWidget(self.black_point_slider)
        white_label = QLabel("White:")
        white_label.setStyleSheet("font-size: 8pt; color: black; background: transparent;")
        levels_layout.addWidget(white_label)
        self.white_point_slider = QSlider(Qt.Horizontal)
        self.white_point_slider.setRange(0, 255)
        self.white_point_slider.setValue(255)
        self.white_point_slider.setFixedWidth(70)
        self.white_point_slider.valueChanged.connect(self.levels_slider_changed)
        levels_layout.addWidget(self.white_point_slider)
        luminance_container.addLayout(levels_layout)
        histograms_container.addLayout(luminance_container)

        rgb_histogram_container = QVBoxLayout()
        rgb_histogram_container.setSpacing(5)
        self.histogram_canvas = HistogramCanvas(self, width=2.5, height=2, dpi=80)
        self.histogram_canvas.setFixedSize(150, 120)
        rgb_histogram_container.addWidget(self.histogram_canvas)
        histograms_container.addLayout(rgb_histogram_container)

        histograms_section = QVBoxLayout()
        histograms_section.addLayout(histograms_container)
        histograms_section.addStretch()
        slider_columns.addLayout(histograms_section)

        # RGB color sliders
        rgb_section = QVBoxLayout()
        rgb_section.setSpacing(10)
        rgb_label = QLabel("RGB Black Point:")
        rgb_label.setAlignment(Qt.AlignCenter)
        rgb_label.setStyleSheet(f"color: black; background: transparent; "
                                f"font-family: '{self.digital_font_family}';")
        rgb_section.addWidget(rgb_label)
        rgb_sliders_container = QWidget()
        rgb_sliders_container.setStyleSheet("background: transparent;")
        rgb_sliders_layout = QVBoxLayout(rgb_sliders_container)
        rgb_sliders_layout.setSpacing(8)
        rgb_sliders_layout.setContentsMargins(0, 0, 0, 0)

        # Red slider
        red_layout = QHBoxLayout()
        red_label = QLabel("R:")
        red_label.setStyleSheet("color: #C00000; background: transparent;")
        red_layout.addWidget(red_label)
        self.red_slider = QSlider(Qt.Horizontal)
        self.red_slider.setRange(0, 255)
        self.red_slider.setValue(0)
        self.red_slider.setFixedWidth(200)
        self.red_slider.valueChanged.connect(self.update_color_from_sliders)
        self.red_slider.setStyleSheet("background: transparent;")
        self.red_slider.setCursor(QCursor(Qt.PointingHandCursor))
        red_layout.addWidget(self.red_slider)
        self.red_value_label = QLabel("0")
        self.red_value_label.setFixedWidth(30)
        self.red_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.red_value_label.setStyleSheet("color: #C00000; background: transparent;")
        red_layout.addWidget(self.red_value_label)
        rgb_sliders_layout.addLayout(red_layout)

        # Green slider
        green_layout = QHBoxLayout()
        green_label = QLabel("G:")
        green_label.setStyleSheet("color: #00C000; background: transparent;")
        green_layout.addWidget(green_label)
        self.green_slider = QSlider(Qt.Horizontal)
        self.green_slider.setRange(0, 255)
        self.green_slider.setValue(0)
        self.green_slider.setFixedWidth(200)
        self.green_slider.valueChanged.connect(self.update_color_from_sliders)
        self.green_slider.setStyleSheet("background: transparent;")
        self.green_slider.setCursor(QCursor(Qt.PointingHandCursor))
        green_layout.addWidget(self.green_slider)
        self.green_value_label = QLabel("0")
        self.green_value_label.setFixedWidth(30)
        self.green_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.green_value_label.setStyleSheet("color: #00C000; background: transparent;")
        green_layout.addWidget(self.green_value_label)
        rgb_sliders_layout.addLayout(green_layout)

        # Blue slider
        blue_layout = QHBoxLayout()
        blue_label = QLabel("B:")
        blue_label.setStyleSheet("color: #0000C0; background: transparent;")
        blue_layout.addWidget(blue_label)
        self.blue_slider = QSlider(Qt.Horizontal)
        self.blue_slider.setRange(0, 255)
        self.blue_slider.setValue(0)
        self.blue_slider.setFixedWidth(200)
        self.blue_slider.valueChanged.connect(self.update_color_from_sliders)
        self.blue_slider.setStyleSheet("background: transparent;")
        self.blue_slider.setCursor(QCursor(Qt.PointingHandCursor))
        blue_layout.addWidget(self.blue_slider)
        self.blue_value_label = QLabel("0")
        self.blue_value_label.setFixedWidth(30)
        self.blue_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.blue_value_label.setStyleSheet("color: #0000C0; background: transparent;")
        blue_layout.addWidget(self.blue_value_label)
        rgb_sliders_layout.addLayout(blue_layout)

        rgb_section.addWidget(rgb_sliders_container)
        slider_columns.addLayout(rgb_section)

        # Image adjustments section (Hue, Contrast, Shadows, Saturation, Brightness, Highlights)
        adjust_section = QVBoxLayout()
        adjust_section.setSpacing(10)
        adjust_label = QLabel("Image Adjustments:")
        adjust_label.setAlignment(Qt.AlignCenter)
        adjust_label.setStyleSheet(f"color: black; background: transparent; "
                                   f"font-family: '{self.digital_font_family}';")
        adjust_section.addWidget(adjust_label)
        adjust_grid = QGridLayout()
        adjust_grid.setSpacing(8)
        adjust_grid.setContentsMargins(0, 0, 0, 0)

        # Column 1: Hue, Contrast, Shadows
        hue_layout = QHBoxLayout()
        hue_label = QLabel("Hue:")
        hue_label.setStyleSheet("color: black; background: transparent;")
        hue_layout.addWidget(hue_label)
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(0, 360)
        self.hue_slider.setValue(0)
        self.hue_slider.setFixedWidth(130)
        self.hue_slider.valueChanged.connect(self.update_image_adjustments)
        self.hue_slider.setStyleSheet("background: transparent;")
        self.hue_slider.setCursor(QCursor(Qt.PointingHandCursor))
        hue_layout.addWidget(self.hue_slider)
        self.hue_value_label = QLabel("0")
        self.hue_value_label.setFixedWidth(30)
        self.hue_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hue_value_label.setStyleSheet("color: black; background: transparent;")
        hue_layout.addWidget(self.hue_value_label)
        hue_container = QWidget()
        hue_container.setStyleSheet("background: transparent;")
        hue_container.setLayout(hue_layout)
        adjust_grid.addWidget(hue_container, 0, 0)

        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Con:")
        contrast_label.setStyleSheet("color: black; background: transparent;")
        contrast_layout.addWidget(contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(130)
        self.contrast_slider.valueChanged.connect(self.update_image_adjustments)
        self.contrast_slider.setStyleSheet("background: transparent;")
        self.contrast_slider.setCursor(QCursor(Qt.PointingHandCursor))
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel("100")
        self.contrast_value_label.setFixedWidth(30)
        self.contrast_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.contrast_value_label.setStyleSheet("color: black; background: transparent;")
        contrast_layout.addWidget(self.contrast_value_label)
        contrast_container = QWidget()
        contrast_container.setStyleSheet("background: transparent;")
        contrast_container.setLayout(contrast_layout)
        adjust_grid.addWidget(contrast_container, 1, 0)

        shadows_layout = QHBoxLayout()
        shadows_label = QLabel("Sha:")
        shadows_label.setStyleSheet("color: #303030; background: transparent;")
        shadows_layout.addWidget(shadows_label)
        self.shadows_slider = QSlider(Qt.Horizontal)
        self.shadows_slider.setRange(0, 200)
        self.shadows_slider.setValue(100)
        self.shadows_slider.setFixedWidth(130)
        self.shadows_slider.valueChanged.connect(self.update_image_adjustments)
        self.shadows_slider.setStyleSheet("background: transparent;")
        self.shadows_slider.setCursor(QCursor(Qt.PointingHandCursor))
        shadows_layout.addWidget(self.shadows_slider)
        self.shadows_value_label = QLabel("100")
        self.shadows_value_label.setFixedWidth(30)
        self.shadows_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.shadows_value_label.setStyleSheet("color: #303030; background: transparent;")
        shadows_layout.addWidget(self.shadows_value_label)
        shadows_container = QWidget()
        shadows_container.setStyleSheet("background: transparent;")
        shadows_container.setLayout(shadows_layout)
        adjust_grid.addWidget(shadows_container, 2, 0)

        # Column 2: Saturation, Brightness, Highlights
        saturation_layout = QHBoxLayout()
        saturation_label = QLabel("Sat:")
        saturation_label.setStyleSheet("color: black; background: transparent;")
        saturation_layout.addWidget(saturation_label)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_slider.setFixedWidth(130)
        self.saturation_slider.valueChanged.connect(self.update_image_adjustments)
        self.saturation_slider.setStyleSheet("background: transparent;")
        self.saturation_slider.setCursor(QCursor(Qt.PointingHandCursor))
        saturation_layout.addWidget(self.saturation_slider)
        self.saturation_value_label = QLabel("100")
        self.saturation_value_label.setFixedWidth(30)
        self.saturation_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.saturation_value_label.setStyleSheet("color: black; background: transparent;")
        saturation_layout.addWidget(self.saturation_value_label)
        saturation_container = QWidget()
        saturation_container.setStyleSheet("background: transparent;")
        saturation_container.setLayout(saturation_layout)
        adjust_grid.addWidget(saturation_container, 0, 1)

        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Bri:")
        brightness_label.setStyleSheet("color: black; background: transparent;")
        brightness_layout.addWidget(brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setFixedWidth(130)
        self.brightness_slider.valueChanged.connect(self.update_image_adjustments)
        self.brightness_slider.setStyleSheet("background: transparent;")
        self.brightness_slider.setCursor(QCursor(Qt.PointingHandCursor))
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel("100")
        self.brightness_value_label.setFixedWidth(30)
        self.brightness_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.brightness_value_label.setStyleSheet("color: black; background: transparent;")
        brightness_layout.addWidget(self.brightness_value_label)
        brightness_container = QWidget()
        brightness_container.setStyleSheet("background: transparent;")
        brightness_container.setLayout(brightness_layout)
        adjust_grid.addWidget(brightness_container, 1, 1)

        highlights_layout = QHBoxLayout()
        highlights_label = QLabel("Hig:")
        highlights_label.setStyleSheet("color: #E0E0E0; background: transparent;")
        highlights_layout.addWidget(highlights_label)
        self.highlights_slider = QSlider(Qt.Horizontal)
        self.highlights_slider.setRange(0, 200)
        self.highlights_slider.setValue(100)
        self.highlights_slider.setFixedWidth(130)
        self.highlights_slider.valueChanged.connect(self.update_image_adjustments)
        self.highlights_slider.setStyleSheet("background: transparent;")
        self.highlights_slider.setCursor(QCursor(Qt.PointingHandCursor))
        highlights_layout.addWidget(self.highlights_slider)
        self.highlights_value_label = QLabel("100")
        self.highlights_value_label.setFixedWidth(30)
        self.highlights_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.highlights_value_label.setStyleSheet("color: #E0E0E0; background: transparent;")
        highlights_layout.addWidget(self.highlights_value_label)
        highlights_container = QWidget()
        highlights_container.setStyleSheet("background: transparent;")
        highlights_container.setLayout(highlights_layout)
        adjust_grid.addWidget(highlights_container, 2, 1)

        adjust_grid_widget = QWidget()
        adjust_grid_widget.setStyleSheet("background: transparent;")
        adjust_grid_widget.setLayout(adjust_grid)
        adjust_section.addWidget(adjust_grid_widget)
        slider_columns.addLayout(adjust_section)

        sliders_container.addLayout(slider_columns)
        layout.addLayout(sliders_container)

        # Preset controls and color picker
        preset_controls = QHBoxLayout()
        preset_controls.setSpacing(15)
        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet(f"color: black; background: transparent; "
                                   f"font-family: '{self.digital_font_family}';")
        preset_controls.addWidget(preset_label)
        self.preset_buttons = {}
        for preset in self.presets:
            btn = QPushButton(preset)
            btn.setFixedSize(60, 60)
            btn.clicked.connect(lambda checked, p=preset: self.set_preset(p))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: black;
                    border: 1px solid black;
                    padding: 6px;
                    font-size: 12px;
                    text-align: center;
                    font-family: '{self.digital_font_family}';
                }}
                QPushButton:hover {{
                    background-color: black;
                    color: white;
                }}
                QPushButton:pressed {{
                    background-color: #333333;
                    color: white;
                }}
            """)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            if preset == "Auto":
                btn.setStyleSheet(btn.styleSheet().replace("border: 1px solid black;",
                                                           "border: 2px solid black;"))
            self.preset_buttons[preset] = btn
            preset_controls.addWidget(btn)
        preset_controls.addStretch()

        color_frame = QFrame()
        color_frame.setStyleSheet("background: transparent;")
        color_layout = QHBoxLayout(color_frame)
        color_layout.setSpacing(5)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_label = QLabel("Base Color:")
        color_label.setStyleSheet(f"color: black; background: transparent; "
                                  f"font-family: '{self.digital_font_family}';")
        color_layout.addWidget(color_label)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(40, 40)
        self.color_preview.setStyleSheet("border: 1px solid black;")
        color_layout.addWidget(self.color_preview)
        pick_btn = QPushButton("Pick")
        pick_btn.setFixedSize(60, 60)
        pick_btn.clicked.connect(self.pick_color)
        pick_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: black;
                border: 1px solid black;
                padding: 6px;
                font-size: 12px;
                text-align: center;
                font-family: '{self.digital_font_family}';
            }}
            QPushButton:hover {{
                background-color: black;
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #333333;
                color: white;
            }}
        """)
        pick_btn.setCursor(QCursor(Qt.PointingHandCursor))
        color_layout.addWidget(pick_btn)
        preset_controls.addWidget(color_frame)
        layout.addLayout(preset_controls)

    def style_button(self, button):
        """Apply a consistent style to a button."""
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: black;
                border: 1px solid black;
                padding: 8px 15px;
                font-size: 12px;
                font-family: '{self.digital_font_family}';
            }}
            QPushButton:hover {{
                background-color: black;
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #333333;
                color: white;
            }}
        """)
        button.setCursor(QCursor(Qt.PointingHandCursor))

    def load_image(self):
        """Load an image from file. Supports PNG, JPG, TIFF, and DNG."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp *.tif *.dng)"
        )
        if file_name:
            ext = os.path.splitext(file_name)[1].lower()
            if ext == ".dng":
                if rawpy is None:
                    print("rawpy module not available. Cannot load DNG files.")
                    return
                try:
                    with rawpy.imread(file_name) as raw:
                        rgb = raw.postprocess()
                        self.image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print("Error reading DNG file:", e)
                    return
            else:
                self.image = cv2.imread(file_name)
            self.auto_detected_color = detect_base_color(self.image)
            self.base_color = None
            self.current_preset = "Auto"
            self.update_preset_buttons()
            self.display_negative()
            self.update_color_preview()
            self.update_sliders_from_color()
            self.reset_adjustment_sliders()
            self.positive_label.clear()
        if self.histogram_canvas is not None:
            self.histogram_canvas.update_histogram(self.image, mode='rgb')
        if self.luminance_histogram is not None:
            self.luminance_histogram.update_histogram(self.image, mode='luminance')

    def pick_color(self):
        """Open a color picker and update the base color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.base_color = (color.red(), color.green(), color.blue())
            self.current_preset = "Auto"
            self.update_preset_buttons()
            self.update_color_preview()
            self.update_sliders_from_color()
            self.process_image()

    def reset_adjustment_sliders(self):
        """Reset all adjustment sliders to their default values."""
        self.slider_update_in_progress = True
        self.hue_slider.setValue(0)
        self.saturation_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(100)
        self.shadows_slider.setValue(100)
        self.highlights_slider.setValue(100)
        if hasattr(self, 'black_point_slider'):
            self.black_point_slider.setValue(0)
        if hasattr(self, 'white_point_slider'):
            self.white_point_slider.setValue(255)
        self.hue_value = 0
        self.saturation_value = 100
        self.contrast_value = 100
        self.brightness_value = 100
        self.shadows_value = 100
        self.highlights_value = 100
        self.hue_value_label.setText("0")
        self.saturation_value_label.setText("100")
        self.contrast_value_label.setText("100")
        self.brightness_value_label.setText("100")
        self.shadows_value_label.setText("100")
        self.highlights_value_label.setText("100")
        self.slider_update_in_progress = False

    def update_sliders_from_color(self):
        """Update the RGB sliders to match the current base color."""
        current_color = self.get_current_color()
        if current_color:
            self.slider_update_in_progress = True
            r, g, b = current_color
            self.red_slider.setValue(r)
            self.red_value_label.setText(str(r))
            self.green_slider.setValue(g)
            self.green_value_label.setText(str(g))
            self.blue_slider.setValue(b)
            self.blue_value_label.setText(str(b))
            self.slider_update_in_progress = False

    def update_color_from_sliders(self):
        """Update the base color based on the RGB slider values."""
        if not self.slider_update_in_progress:
            r = self.red_slider.value()
            g = self.green_slider.value()
            b = self.blue_slider.value()
            self.red_value_label.setText(str(r))
            self.green_value_label.setText(str(g))
            self.blue_value_label.setText(str(b))
            self.base_color = (r, g, b)
            self.current_preset = "Auto"
            self.update_preset_buttons()
            self.update_color_preview()
            if self.image is not None:
                self.process_image()

    def update_image_adjustments(self):
        """Update adjustment values and process the image."""
        if not self.slider_update_in_progress:
            self.hue_value = self.hue_slider.value()
            self.saturation_value = self.saturation_slider.value()
            self.contrast_value = self.contrast_slider.value()
            self.brightness_value = self.brightness_slider.value()
            self.shadows_value = self.shadows_slider.value()
            self.highlights_value = self.highlights_slider.value()
            self.hue_value_label.setText(str(self.hue_value))
            self.saturation_value_label.setText(str(self.saturation_value))
            self.contrast_value_label.setText(str(self.contrast_value))
            self.brightness_value_label.setText(str(self.brightness_value))
            self.shadows_value_label.setText(str(self.shadows_value))
            self.highlights_value_label.setText(str(self.highlights_value))
            if self.image is not None and self.processed_full_resolution is not None:
                self.process_image()

    def levels_slider_changed(self):
        """Process the image when the levels sliders change."""
        if self.image is not None:
            self.process_image()

    def adjust_levels(self, img):
        """Apply a levels adjustment based on black and white slider values.

        Args:
            img (ndarray): Input image.

        Returns:
            ndarray: Levels-adjusted image.
        """
        black = self.black_point_slider.value()
        white = self.white_point_slider.value()
        if white <= black:
            return img
        img_float = img.astype(np.float32)
        img_float = (img_float - black) / (white - black)
        img_float = np.clip(img_float, 0, 1) * 255
        return img_float.astype(np.uint8)

    def get_current_color(self):
        """Determine the current base color."""
        if self.current_preset == "Auto":
            if self.base_color is not None:
                return self.base_color
            elif self.auto_detected_color is not None:
                return [int(x) for x in self.auto_detected_color]
            else:
                return (0, 0, 0)
        else:
            return self.presets[self.current_preset]

    def image_to_pixmap(self, img):
        """Convert an OpenCV BGR image to a QPixmap."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def display_negative(self):
        """Display the original (negative) image."""
        if self.image is not None:
            pixmap = self.image_to_pixmap(self.image)
            width = max(300, self.negative_label.width())
            height = max(300, self.negative_label.height())
            self.negative_label.setPixmap(pixmap.scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.negative_label.adjustSize()

    def update_color_preview(self):
        """Update the color preview widget based on the current base color."""
        current_color = self.get_current_color()
        if current_color:
            color = QColor(*current_color)
            self.color_preview.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid black;"
            )

    def set_preset(self, preset):
        """Set the preset and update the image accordingly."""
        self.current_preset = preset
        self.base_color = None if preset == "Auto" else self.presets[preset]
        self.update_preset_buttons()
        self.update_color_preview()
        self.update_sliders_from_color()
        if self.image is not None:
            self.process_image()
        else:
            self.positive_label.clear()

    def update_preset_buttons(self):
        """Update the visual appearance of preset buttons."""
        for preset, button in self.preset_buttons.items():
            if preset == self.current_preset:
                button.setStyleSheet(button.styleSheet().replace(
                    "border: 1px solid black;", "border: 2px solid black;"
                ))
            else:
                button.setStyleSheet(button.styleSheet().replace(
                    "border: 2px solid black;", "border: 1px solid black;"
                ))

    def process_image(self):
        """Process the image with the current adjustments and display it."""
        if self.image is None:
            return

        if self.current_preset == "Auto":
            processed = (convert_negative(self.image, self.base_color)
                         if self.base_color is not None
                         else convert_negative(self.image,
                                                 self.auto_detected_color))
        else:
            processed = convert_negative(self.image, self.presets[self.current_preset])

        self.processed_full_resolution = apply_adjustments(
            processed,
            hue=self.hue_value,
            saturation=self.saturation_value,
            contrast=self.contrast_value,
            brightness=self.brightness_value,
            shadows=self.shadows_value,
            highlights=self.highlights_value
        )
        self.processed_full_resolution = self.adjust_levels(self.processed_full_resolution)
        pixmap = self.image_to_pixmap(self.processed_full_resolution)
        width = max(300, self.positive_label.width())
        height = max(300, self.positive_label.height())
        self.positive_label.setPixmap(pixmap.scaled(
            width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.positive_label.adjustSize()
        if self.histogram_canvas is not None:
            self.histogram_canvas.update_histogram(self.processed_full_resolution, mode='rgb')
        if self.luminance_histogram is not None:
            self.luminance_histogram.update_histogram(self.processed_full_resolution, mode='luminance')

    def save_image(self):
        """Save the processed image to a file."""
        if self.processed_full_resolution is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Positive", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if file_name:
            cv2.imwrite(file_name, self.processed_full_resolution)

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if hasattr(self, 'image') and self.image is not None:
            self.display_negative()
            if self.processed_full_resolution is not None:
                self.process_image()


def main():
    """Entry point for the application."""
    app = QApplication(sys.argv)
    window = NegativeConverter()
    window.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
