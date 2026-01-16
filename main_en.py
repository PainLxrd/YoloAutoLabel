# main.py
import sys
import os
from PyQt5.QtWidgets import ( QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog, QTextEdit, QMessageBox, QSlider, QGroupBox, QProgressBar, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage
import threading
from auto_annotator_en import run_auto_annotation, preview_detection, get_classes

class AutoLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Auto-Annotation Tool v2.0")
        self.resize(950, 800)
        self.model_dir = "models"
        self.current_preview_pixmap = None
        self.image_files = [] # List of current images
        self.current_image_index = 0 # Current preview index
        self.img_dir = "" # Path to current image directory
        self.selected_classes = [] # List of user-selected class names
        self.all_model_classes = [] # All class names from the current model
        self.init_ui()
        self.preview_debounce_timer = QTimer()
        self.preview_debounce_timer.setSingleShot(True)
        self.preview_debounce_timer.timeout.connect(self._do_preview_in_thread)

    def load_local_models(self, model_dir="models"):
        """Load all .pt files from the models directory"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            return []
        files = [f for f in os.listdir(model_dir) if f.lower().endswith('.pt')]
        return sorted(files)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # === Top Level: Left-Right Split (Fixed Width) ===
        main_split = QHBoxLayout()
        main_split.setSpacing(15)
        main_split.setContentsMargins(15, 15, 15, 15)

        # --- Left Panel (Fixed Width) ---
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # === Model Selection ===
        local_models = self.load_local_models(self.model_dir)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Select a model")
        self.model_combo.setItemData(0, "", Qt.UserRole)
        if local_models:
            self.model_combo.addItems(local_models)
            self.model_combo.setEnabled(True)
        else:
            self.model_combo.setEnabled(False)
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        model_layout.addWidget(self.model_combo)
        left_layout.addLayout(model_layout)

        # === Class Selection ===
        class_group = QGroupBox("Select Classes to Annotate")
        class_layout = QVBoxLayout()
        class_layout.setContentsMargins(10, 10, 10, 10)
        self.class_list_widget = QListWidget()
        self.class_list_widget.setMinimumHeight(200)
        class_layout.addWidget(self.class_list_widget)
        button_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Invert Selection")
        self.select_all_btn.setStyleSheet("font-size: 10pt; padding: 4px;")
        self.select_none_btn.setStyleSheet("font-size: 10pt; padding: 4px;")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.select_none_btn.clicked.connect(self.select_inverse_classes)
        button_row.addWidget(self.select_all_btn)
        button_row.addWidget(self.select_none_btn)
        button_row.addStretch()
        class_layout.addLayout(button_row)
        class_group.setLayout(class_layout)
        left_layout.addWidget(class_group)

        # === Image Directory ===
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Image Directory:"))
        self.img_dir_edit = QLineEdit()
        self.img_dir_edit.setPlaceholderText("Select images to annotate")
        self.img_dir_btn = QPushButton("Browse")
        self.img_dir_btn.clicked.connect(lambda: self.select_directory(self.img_dir_edit))
        img_layout.addWidget(self.img_dir_edit)
        img_layout.addWidget(self.img_dir_btn)
        left_layout.addLayout(img_layout)

        # === Label Directory ===
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label Directory:"))
        self.label_dir_edit = QLineEdit()
        self.label_dir_edit.setPlaceholderText("Select where to save labels")
        self.label_dir_btn = QPushButton("Browse")
        self.label_dir_btn.clicked.connect(lambda: self.select_directory(self.label_dir_edit))
        label_layout.addWidget(self.label_dir_edit)
        label_layout.addWidget(self.label_dir_btn)
        left_layout.addLayout(label_layout)

        # === Confidence Slider ===
        conf_group = QGroupBox()
        conf_layout = QHBoxLayout()
        conf_layout.setContentsMargins(10, 5, 10, 5)
        self.conf_label = QLabel("Confidence: 0.25", self)
        self.conf_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.conf_label, 0)
        conf_layout.addWidget(self.conf_slider, 1)
        conf_group.setLayout(conf_layout)
        left_layout.addWidget(conf_group)

        # === Button Area ===
        btn_layout = QHBoxLayout()
        self.preview_btn = QPushButton("🔍 Load Preview")
        self.preview_btn.clicked.connect(self.load_and_preview)
        self.start_btn = QPushButton("🚀 Start Auto-Annotation")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.start_btn)
        left_layout.addLayout(btn_layout)

        # === Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # === Log Area ===
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(130)
        left_layout.addWidget(self.log_text)
        left_panel.setLayout(left_layout)

        # --- Right Panel (Fixed Width) ---
        right_panel = QWidget()
        right_panel.setFixedWidth(900)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # === Preview Control Buttons ===
        control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.image_info_label)
        control_layout.addWidget(self.next_btn)
        right_layout.addLayout(control_layout)

        # === Preview Display Area ===
        self.preview_label = QLabel("Click 'Load & Preview Images' to see detection results")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            border: 1px solid #cbd5e1;
            background-color: #f8fafc;
            border-radius: 10px;
            font-size: 14pt;
            color: #718096;
            min-height: 500px;
        """)
        right_layout.addWidget(self.preview_label, 1)
        right_panel.setLayout(right_layout)

        # === Combine Panels + Footer ===
        main_split.addWidget(left_panel)
        main_split.addWidget(right_panel)

        # Create main vertical layout: panels + footer
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 5)
        main_layout.addLayout(main_split)

        # Add footer
        footer_label = QLabel("Author: YouLuoYuan TuBoShu | QQ Group: 307531422")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: gray; font-size: 9pt;")
        footer_label.setFixedHeight(15)
        main_layout.addWidget(footer_label)
        central.setLayout(main_layout)

        # === Global Styles ===
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7fafc;
            }
            QGroupBox {
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 12px;
                margin-top: 0px;
                background-color: #ffffff;
            }
            QLabel {
                font-size: 12pt;
                color: #2d3748;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QPushButton {
                background-color: #4299e1;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                min-width: 90px;
                font-size: 12pt;
                color: white;
                font-weight: 600;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:pressed {
                background-color: #2b6cb0;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4299e1;
                border: 2px solid #ffffff;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #3182ce;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
                font-size: 11pt;
                font-family: "Microsoft YaHei", "Consolas", monospace;
                color: #4a5568;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cbd5e0;
                border-radius: 8px;
                background: #ffffff;
                font-size: 11pt;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #cbd5e0;
                border-radius: 8px;
                background: white;
                font-size: 11pt;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QProgressBar {
                border: 1px solid #cbd5e0;
                border-radius: 8px;
                text-align: center;
                font-size: 10pt;
                color: #4a5568;
                background-color: #f8fafc;
            }
            QProgressBar::chunk {
                background-color: #4299e1;
                border-radius: 7px;
            }
        """)

        # Initialize button states
        self.preview_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        # Set window size
        total_width = 400 + 900 + 50
        self.setMaximumWidth(total_width)
        self.resize(total_width, 720)
        self.setMinimumSize(total_width, 600)

    def update_conf_label(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"Confidence: {conf:.2f}")

    def on_confidence_changed(self, value):
        """On confidence change, start debounce timer to avoid frequent inference"""
        if self.image_files:
            self.preview_debounce_timer.start(150)

    def _clear_class_checkboxes(self):
        self.log_text.append("🔧 Clearing class checkboxes...")
        self.class_list_widget.clear()
        self.all_model_classes = []
        self.selected_classes = []

    def on_model_change(self, text):
        self.log_text.append(f"🔄 on_model_change called, current selection: '{text}'")
        if text in ("Select a model", "(No models available)", ""):
            self._clear_class_checkboxes()
            self.preview_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.log_text.append("⚠️ No valid model selected, disabling buttons")
            return

        self._clear_class_checkboxes()
        try:
            model_path = os.path.join(self.model_dir, text)
            self.log_text.append(f"📂 Attempting to load model: {model_path}")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Get classes
            raw_classes = get_classes(model_path)
            if isinstance(raw_classes, dict):
                max_key = max(raw_classes.keys())
                self.all_model_classes = [raw_classes[i] for i in range(max_key + 1)]
            else:
                self.all_model_classes = raw_classes
            self.log_text.append(
                f"📚 Classes from model: {self.all_model_classes} (Total: {len(self.all_model_classes)})")

            if not self.all_model_classes:
                self.log_text.append("❗ Warning: Model returned an empty class list!")
                self.preview_btn.setEnabled(False)
                self.start_btn.setEnabled(False)
                return

            # Populate QListWidget with checkable items
            self.class_list_widget.clear()
            self.class_list_widget.itemChanged.connect(self.update_selected_classes)
            for cls_name in self.all_model_classes:
                item = QListWidgetItem(cls_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked) # Default to checked
                self.class_list_widget.addItem(item)

            # Manually trigger an update
            self.update_selected_classes()
            self.log_text.append(f"✅ Successfully loaded {len(self.all_model_classes)} classes into the list")

            # Enable buttons
            self.preview_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.log_text.append("🟢 Buttons enabled, awaiting user action")

        except Exception as e:
            error_msg = f"💥 Error in on_model_change: {e}"
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            self._clear_class_checkboxes()
            self.preview_btn.setEnabled(False)
            self.start_btn.setEnabled(False)

    def on_class_selection_changed(self):
        self.update_selected_classes()

    def update_selected_classes(self):
        self.selected_classes = []
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.selected_classes.append(item.text())
        self.log_text.append(f"[DEBUG] Currently selected classes: {self.selected_classes}")

    def select_all_classes(self):
        """Select all classes"""
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            item.setCheckState(Qt.Checked)
        self.update_selected_classes()

    def select_inverse_classes(self):
        """Invert selection"""
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            current = item.checkState()
            item.setCheckState(Qt.Unchecked if current == Qt.Checked else Qt.Checked)
        self.update_selected_classes()

    def select_directory(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    def get_selected_model(self):
        text = self.model_combo.currentText()
        if text in ("Select a model", "(No models available)", ""):
            return None
        model_path = os.path.join(self.model_dir, text)
        return model_path if os.path.isfile(model_path) else None

    def get_confidence(self):
        return self.conf_slider.value() / 100.0

    def load_and_preview(self):
        model_path = self.get_selected_model()
        img_dir = self.img_dir_edit.text()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "Error", "Please place a .pt model file in the models/ directory and select it from the dropdown!")
            return
        if not img_dir or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "Error", "Please select an image directory!")
            return

        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
        if not image_files:
            QMessageBox.warning(self, "Info", "No valid images found in the directory!")
            return

        self.image_files = sorted(image_files)
        self.img_dir = img_dir
        self.current_image_index = 0
        self.update_preview()
        self.log_text.append(f"✅ Successfully loaded {len(self.image_files)} images, current: {self.image_files[0]}")

    def update_preview(self):
        if not self.image_files:
            return
        current_img = self.image_files[self.current_image_index]
        img_path = os.path.join(self.img_dir, current_img)
        try:
            annotated = preview_detection(
                self.get_selected_model(),
                img_path,
                self.get_confidence(),
                selected_classes=self.selected_classes or None
            )
            if annotated is None:
                raise Exception("Failed to read image")

            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            q_img = QImage(annotated.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.current_preview_pixmap = pixmap
            self.update_preview_display()
            self.image_info_label.setText(f"{self.current_image_index + 1} / {len(self.image_files)} - {current_img}")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))
            self.log_text.append(f"❌ Preview failed: {e}")

    def update_preview_display(self):
        if self.current_preview_pixmap:
            label_size = self.preview_label.size()
            scaled = self.current_preview_pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.current_preview_pixmap:
            self.update_preview_display()
        super().resizeEvent(event)

    def prev_image(self):
        if self.image_files:
            self.current_image_index = max(0, self.current_image_index - 1)
            self.update_preview()

    def next_image(self):
        if self.image_files:
            self.current_image_index = min(len(self.image_files) - 1, self.current_image_index + 1)
            self.update_preview()

    def start_annotation(self):
        model_path = self.get_selected_model()
        img_dir = self.img_dir_edit.text()
        label_dir = self.label_dir_edit.text()
        conf = self.get_confidence()

        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "Error", "Please place a .pt model file in the models/ directory and select it from the dropdown!")
            return
        if not img_dir or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "Error", "Please select a valid image directory!")
            return
        if not label_dir:
            QMessageBox.warning(self, "Error", "Please select a label output directory!")
            return

        self.log_text.append(f"[DEBUG] Starting auto-annotation with selected_classes = {self.selected_classes}")
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.append(f"\n▶ Starting auto-annotation (confidence={conf:.2f})...\n")

        def run_in_thread():
            try:
                total = 0
                for processed, total in run_auto_annotation(
                    model_path, img_dir, label_dir, conf, selected_classes=self.selected_classes
                ):
                    QMetaObject.invokeMethod(self, "update_progress", Qt.QueuedConnection,
                                            Q_ARG(int, processed), Q_ARG(int, total))
                QMetaObject.invokeMethod(self, "_on_finished", Qt.QueuedConnection, Q_ARG(int, total))
            except Exception as e:
                QMetaObject.invokeMethod(self, "_on_error", Qt.QueuedConnection, Q_ARG(str, str(e)))

        threading.Thread(target=run_in_thread, daemon=True).start()

    def _do_preview_in_thread(self):
        """Start a thread for preview detection"""
        if not self.image_files:
            return
        model_path = self.get_selected_model()
        img_path = os.path.join(self.img_dir, self.image_files[self.current_image_index])
        conf = self.get_confidence()
        self.conf_slider.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        def run_preview():
            try:
                annotated = preview_detection(
                    model_path, img_path, conf, selected_classes=self.selected_classes
                )
                QMetaObject.invokeMethod(
                    self, "_on_preview_ready", Qt.QueuedConnection,
                    Q_ARG(object, annotated), Q_ARG(str, self.image_files[self.current_image_index])
                )
            except Exception as e:
                QMetaObject.invokeMethod(
                    self, "_on_preview_error", Qt.QueuedConnection, Q_ARG(str, str(e))
                )

        threading.Thread(target=run_preview, daemon=True).start()

    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)

    @pyqtSlot(int)
    def _on_finished(self, total):
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Finished", f"Auto-annotation complete! Processed {total} images.\nclasses.txt has been generated.")
        self.log_text.append(f"\n✅ Auto-annotation finished! Processed {total} images, classes.txt generated.")

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", msg)
        self.log_text.append(f"\n❌ Error: {msg}")

    @pyqtSlot(object, str)
    def _on_preview_ready(self, annotated, filename):
        """Receive preview result in main thread and update UI"""
        self.conf_slider.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        if annotated is None:
            self.log_text.append("❌ Preview returned an empty image")
            return
        try:
            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            q_img = QImage(annotated.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.current_preview_pixmap = pixmap
            self.update_preview_display()
            self.image_info_label.setText(f"{self.current_image_index + 1} / {len(self.image_files)} - {filename}")
        except Exception as e:
            self.log_text.append(f"❌ Image conversion failed: {e}")

    @pyqtSlot(str)
    def _on_preview_error(self, msg):
        self.conf_slider.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.log_text.append(f"❌ Preview error: {msg}")
        QMessageBox.critical(self, "Preview Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoLabelTool()
    window.show()
    sys.exit(app.exec_())