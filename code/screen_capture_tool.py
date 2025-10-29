import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QComboBox,
)
from PyQt5.QtCore import QTimer
import mss
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class ScreenCaptureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.capture_running = False
        self.metadata_list = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_screen)
        self.monitors = self.get_monitors()
        self.init_ui()
        self.setup_styles()

    def setup_styles(self):
        # 기본 버튼 스타일
        self.normal_style = """
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """
        # 녹화중 버튼 스타일
        self.recording_style = """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """
        self.start_button.setStyleSheet(self.normal_style)

    def get_monitors(self):
        with mss.mss() as sct:
            return [
                (f"Monitor {i+1} ({m['width']}x{m['height']})", m)
                for i, m in enumerate(sct.monitors[1:])
            ]

    def get_frontmost_app(self):
        script = 'tell application "System Events" to get name of first process whose frontmost is true'
        result = subprocess.run(
            ["osascript", "-e", script], stdout=subprocess.PIPE, text=True
        )
        return result.stdout.strip()

    def init_ui(self):
        # Create widgets
        task_type_label = QLabel("Task Type (e.g., TOUR, SHOP, STUD):")
        self.task_type_input = QLineEdit()

        task_number_label = QLabel("Task Number (000-999):")
        self.task_number_input = QLineEdit()

        variation_label = QLabel("Variation Number (00-99):")
        self.variation_input = QLineEdit()

        task_name_label = QLabel("Task Name:")
        self.task_name_input = QLineEdit()

        monitor_label = QLabel("Select Monitor:")
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItems([m[0] for m in self.monitors])

        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self.toggle_capture)

        # Create layouts
        main_layout = QVBoxLayout()

        type_layout = QHBoxLayout()
        type_layout.addWidget(task_type_label)
        type_layout.addWidget(self.task_type_input)

        task_layout = QHBoxLayout()
        task_layout.addWidget(task_number_label)
        task_layout.addWidget(self.task_number_input)

        variation_layout = QHBoxLayout()
        variation_layout.addWidget(variation_label)
        variation_layout.addWidget(self.variation_input)

        name_layout = QHBoxLayout()
        name_layout.addWidget(task_name_label)
        name_layout.addWidget(self.task_name_input)

        monitor_layout = QHBoxLayout()
        monitor_layout.addWidget(monitor_label)
        monitor_layout.addWidget(self.monitor_combo)

        # Add all layouts to main layout
        main_layout.addLayout(type_layout)
        main_layout.addLayout(task_layout)
        main_layout.addLayout(variation_layout)
        main_layout.addLayout(name_layout)
        main_layout.addLayout(monitor_layout)
        main_layout.addWidget(self.start_button)

        self.setLayout(main_layout)
        self.setWindowTitle("Screen Capture Tool")
        self.setGeometry(100, 100, 500, 250)

    def validate_inputs(self):
        try:
            task_type = self.task_type_input.text().strip().upper()
            if not task_type:
                raise ValueError("Task type cannot be empty")

            task_num = int(self.task_number_input.text())
            var_num = int(self.variation_input.text())

            if not (0 <= task_num <= 999):
                raise ValueError("Task number must be between 000-999")
            if not (0 <= var_num <= 99):
                raise ValueError("Variation number must be between 00-99")
            if not self.task_name_input.text().strip():
                raise ValueError("Task name cannot be empty")

            return True
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return False

    def create_folder(self):
        task_type = self.task_type_input.text().strip().upper()
        task_num = self.task_number_input.text().zfill(3)
        var_num = self.variation_input.text().zfill(2)
        task_name = self.task_name_input.text().strip()

        folder_id = f"{task_type}_{task_num}_{var_num}"
        self.folder_path = DATASET_IMAGES_DIR / folder_id
        self.folder_path.mkdir(parents=True, exist_ok=True)

        # 세션 정보도 별도 파일로 기록
        info_path = self.folder_path / "_session_info.json"
        session_info = {
            "folder_id": folder_id,
            "task_type": task_type,
            "task_number": task_num,
            "variation_number": var_num,
            "task_name": task_name,
            "created_at": datetime.now().isoformat(),
        }
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=4, ensure_ascii=False)

    def capture_screen(self):
        with mss.mss() as sct:
            monitor_idx = self.monitor_combo.currentIndex()
            monitor = self.monitors[monitor_idx][1]
            screenshot = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            img = img.convert("RGB")
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.jpg"
            file_path = self.folder_path / filename
            img.save(str(file_path), "JPEG", quality=70, optimize=False, subsampling=0)

            # Get frontmost app
            frontmost_app = self.get_frontmost_app()

            # Append metadata
            metadata_entry = {
                "timestamp": timestamp,
                "file_name": filename,
                "frontmost_app": frontmost_app,
            }
            self.metadata_list.append(metadata_entry)

            # Save to _metadata.json
            metadata_path = self.folder_path / "_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.metadata_list, f, indent=4, ensure_ascii=False)

    def toggle_capture(self):
        if not self.capture_running:
            if not self.validate_inputs():
                return

            self.create_folder()
            self.capture_running = True
            self.start_button.setText("Stop Capture")
            self.start_button.setStyleSheet(self.recording_style)
            self.timer.start(1000)  # Start timer with 1 second interval

            # Disable inputs while capturing
            self.task_type_input.setEnabled(False)
            self.task_number_input.setEnabled(False)
            self.variation_input.setEnabled(False)
            self.task_name_input.setEnabled(False)
            self.monitor_combo.setEnabled(False)
        else:
            self.capture_running = False
            self.start_button.setText("Start Capture")
            self.start_button.setStyleSheet(self.normal_style)
            self.timer.stop()

            # Enable inputs after stopping
            self.task_type_input.setEnabled(True)
            self.task_number_input.setEnabled(True)
            self.variation_input.setEnabled(True)
            self.task_name_input.setEnabled(True)
            self.monitor_combo.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    ex = ScreenCaptureApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
