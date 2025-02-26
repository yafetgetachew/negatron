import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QColorDialog, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt

from image_processing import detect_base_color, convert_negative

class NegativeConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEGATRON v0.1")
        self.setWindowFlags(Qt.FramelessWindowHint)  # Remove native window frame (Done)
        self.setStyleSheet("background-color: #f5e6d2; color: #800020;")

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
        layout.setContentsMargins(10, 10, 10, 10)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addStretch()
        close_btn = QPushButton("x")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5e6d2;
                color: #800020;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e0d3bd;
            }
        """)
        header_layout.addWidget(close_btn)
        layout.addLayout(header_layout)

        # Image display frame TODO: Probably enlarge image to fit within boundaries
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: #f5e6d2; border-radius: 5px;")
        image_layout = QHBoxLayout(image_frame)

        self.negative_label = QLabel("Negative")
        self.negative_label.setAlignment(Qt.AlignCenter)
        self.negative_label.setStyleSheet("color: #800020; font-size: 12px;")
        image_layout.addWidget(self.negative_label)

        self.positive_label = QLabel("Positive")
        self.positive_label.setAlignment(Qt.AlignCenter)
        self.positive_label.setStyleSheet("color: #800020; font-size: 12px;")
        image_layout.addWidget(self.positive_label)

        layout.addWidget(image_frame, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_image)
        self.style_button(load_btn)
        controls.addWidget(load_btn)

        # TODO: make buttons more ergonomic, to occupy less space
        self.preset_buttons = {}
        for preset in self.presets:
            btn = QPushButton(preset)
            btn.clicked.connect(lambda checked, p=preset: self.set_preset(p))
            self.style_button(btn)
            if preset == "Auto":
                btn.setStyleSheet(btn.styleSheet() + "background-color: #e0d3bd;")
            self.preset_buttons[preset] = btn
            controls.addWidget(btn)

        color_frame = QFrame()
        color_layout = QHBoxLayout(color_frame)
        color_layout.setSpacing(5)

        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("border: 1px solid #800020; border-radius: 3px;")
        color_layout.addWidget(self.color_preview)

        pick_btn = QPushButton("Pick")
        pick_btn.clicked.connect(self.pick_color)
        self.style_button(pick_btn)
        color_layout.addWidget(pick_btn)

        controls.addWidget(color_frame)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.process_image)
        self.style_button(apply_btn)
        controls.addWidget(apply_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_image)
        self.style_button(save_btn)
        controls.addWidget(save_btn)

        layout.addLayout(controls)

    def style_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #f5e6d2;
                color: #800020;
                border: 1px solid #800020;
                border-radius: 3px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e0d3bd;
            }
            QPushButton:pressed {
                background-color: #d1c4a8;
            }
        """)

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
        return QPixmap.fromImage(q_img).scaled(480, 580, Qt.KeepAspectRatio)

    def display_negative(self):
        if self.image is not None:
            pixmap = self.image_to_pixmap(self.image)
            self.negative_label.setPixmap(pixmap)

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
            f"background-color: {color.name()}; border: 1px solid #800020; border-radius: 3px;"
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
                button.setStyleSheet(button.styleSheet() + "background-color: #e0d3bd;")
            else:
                button.setStyleSheet(button.styleSheet().replace("background-color: #e0d3bd;", ""))

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
        self.positive_label.setPixmap(pixmap)

    def save_image(self):
        if self.processed_full_resolution is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Positive", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if file_name:
            cv2.imwrite(file_name, self.processed_full_resolution)

def main():
    app = QApplication(sys.argv)
    window = NegativeConverter()
    window.showFullScreen()  # Now it starts in full-screen (Should it?)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
