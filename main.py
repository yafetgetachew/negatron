import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QColorDialog, QFrame, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QCursor, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QSize

from image_processing import detect_base_color, convert_negative

class NegativeConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEGATRON v0.1")
        self.setWindowFlags(Qt.FramelessWindowHint)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(script_dir, 'fonts', 'Bowman.ttf')

        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            self.digital_font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            print(f"Error: Could not load digital font from {font_path}. Using system default.")
            self.digital_font_family = QFont().family()

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #808080, stop:1 #b0c4de);
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

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
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

        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: transparent;")
        image_layout = QHBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(20)
        negative_container = QWidget()
        negative_layout = QVBoxLayout(negative_container)
        negative_layout.setContentsMargins(0, 0, 0, 0)

        negative_title = QLabel("Negative")
        negative_title.setAlignment(Qt.AlignCenter)
        negative_title.setStyleSheet(f"""
            color: black;
            font-size: 30px;
            font-weight: bold;
            font-family: '{self.digital_font_family}';
        """)
        negative_layout.addWidget(negative_title)

        self.negative_label = QLabel()
        self.negative_label.setAlignment(Qt.AlignCenter)
        self.negative_label.setMinimumSize(300, 300)
        self.negative_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        negative_layout.addWidget(self.negative_label)

        positive_container = QWidget()
        positive_layout = QVBoxLayout(positive_container)
        positive_layout.setContentsMargins(0, 0, 0, 0)

        positive_title = QLabel("Positive")
        positive_title.setAlignment(Qt.AlignCenter)
        positive_title.setStyleSheet(f"""
            color: black;
            font-size: 30px;
            font-weight: bold;
            font-family: '{self.digital_font_family}';
        """)
        positive_layout.addWidget(positive_title)

        self.positive_label = QLabel()
        self.positive_label.setAlignment(Qt.AlignCenter)
        self.positive_label.setMinimumSize(300, 300)
        self.positive_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        positive_layout.addWidget(self.positive_label)

        image_layout.addWidget(negative_container)
        image_layout.addWidget(positive_container)

        layout.addWidget(image_frame, stretch=1)

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

        preset_controls = QHBoxLayout()
        preset_controls.setSpacing(15)

        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet(f"""
            color: black;
            background: transparent;
            font-family: '{self.digital_font_family}';
        """)
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
                btn.setStyleSheet(btn.styleSheet().replace("border: 1px solid black;", "border: 2px solid black;"))

            self.preset_buttons[preset] = btn
            preset_controls.addWidget(btn)

        preset_controls.addStretch()

        color_frame = QFrame()
        color_frame.setStyleSheet("background: transparent;")
        color_layout = QHBoxLayout(color_frame)
        color_layout.setSpacing(5)
        color_layout.setContentsMargins(0, 0, 0, 0)

        color_label = QLabel("Base Color:")
        color_label.setStyleSheet(f"""
            color: black;
            background: transparent;
            font-family: '{self.digital_font_family}';
        """)
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
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp)"
        )
        if file_name:
            self.image = cv2.imread(file_name)
            self.auto_detected_color = detect_base_color(self.image)
            self.base_color = None
            self.current_preset = "Auto"
            self.update_preset_buttons()
            self.display_negative()
            self.update_color_preview()
            self.positive_label.clear()

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.base_color = (color.red(), color.green(), color.blue())
            self.current_preset = "Auto"
            self.update_preset_buttons()
            self.update_color_preview()

    def image_to_pixmap(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

    def display_negative(self):
        if self.image is not None:
            pixmap = self.image_to_pixmap(self.image)
            width = max(300, self.negative_label.width())
            height = max(300, self.negative_label.height())
            self.negative_label.setPixmap(pixmap.scaled(
                width,
                height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.negative_label.adjustSize()

    def update_color_preview(self):
        if self.current_preset == "Auto":
            if self.base_color is not None:
                color = QColor(*self.base_color)
            elif self.auto_detected_color is not None:
                color = QColor(*[int(x) for x in self.auto_detected_color])
            else:
                color = QColor(0, 0, 0)
        else:
            color = QColor(*self.presets[self.current_preset])
        self.color_preview.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid black;"
        )

    def set_preset(self, preset):
        self.current_preset = preset
        if preset == "Auto":
            self.base_color = None
            self.positive_label.clear()
        else:
            self.base_color = self.presets[preset]
        self.update_preset_buttons()
        self.update_color_preview()

    def update_preset_buttons(self):
        for preset, button in self.preset_buttons.items():
            if preset == self.current_preset:
                button.setStyleSheet(button.styleSheet().replace("border: 1px solid black;", "border: 2px solid black;"))
            else:
                button.setStyleSheet(button.styleSheet().replace("border: 2px solid black;", "border: 1px solid black;"))

    def process_image(self):
        if self.image is None:
            return

        if self.current_preset == "Auto":
            if self.base_color is not None:
                processed = convert_negative(self.image, self.base_color)
            else:
                processed = convert_negative(self.image, self.auto_detected_color)
        else:
            processed = convert_negative(self.image, self.presets[self.current_preset])

        self.processed_full_resolution = processed

        pixmap = self.image_to_pixmap(processed)
        width = max(300, self.positive_label.width())
        height = max(300, self.positive_label.height())
        self.positive_label.setPixmap(pixmap.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.positive_label.adjustSize()

    def save_image(self):
        if self.processed_full_resolution is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Positive", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if file_name:
            cv2.imwrite(file_name, self.processed_full_resolution)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image') and self.image is not None:
            self.display_negative()
            if self.processed_full_resolution is not None:
                self.process_image()

def main():
    app = QApplication(sys.argv)
    window = NegativeConverter()
    window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
