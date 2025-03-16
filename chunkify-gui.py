import sys
import os
import asyncio
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QRadioButton, QLabel, QTextEdit, QFileDialog,
    QDialog, QFormLayout, QLineEdit, QSpinBox, QDialogButtonBox,
    QGroupBox, QScrollArea, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread

import chunkify


class OutputRedirector(QObject):
    """ Redirects print output to a text widget """
    text_output = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if '\n' in text:
            self.text_output.emit(self.buffer)
            self.buffer = ""
        return len(text)

    def flush(self):
        if self.buffer:
            self.text_output.emit(self.buffer)
            self.buffer = ""


class WorkerThread(QThread):
    """ Thread for running the text processing operations """
    finished = pyqtSignal(int)
    progress = pyqtSignal(int, int)  # (current_chunk, total_chunks)

    def __init__(self, api_url, input_path, task, output_path, language, max_chunk_size, api_password):
        super().__init__()
        self.api_url = api_url
        self.input_path = input_path
        self.task = task
        self.output_path = output_path
        self.language = language
        self.max_chunk_size = max_chunk_size
        self.api_password = api_password

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            exit_code = loop.run_until_complete(chunkify.process_file(
                api_url=self.api_url,
                input_path=self.input_path,
                task=self.task,
                output_path=self.output_path,
                language=self.language,
                max_chunk_size=self.max_chunk_size,
                api_password=self.api_password
            ))
            self.finished.emit(exit_code)
        except Exception as e:
            print(f"Error in worker thread: {str(e)}")
            self.finished.emit(1)


class ConfigDialog(QDialog):
    """ Dialog for configuring the API settings """
    
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        
        self.settings = settings or {
            "api_url": "http://localhost:5001",
            "api_password": "",
            "language": "English",
            "max_chunk_size": 4096
        }
        
        self.setWindowTitle("Configuration")
        self.resize(400, 200)
        
        # Create form layout for settings
        layout = QFormLayout(self)
        
        # API URL
        self.api_url_edit = QLineEdit(self.settings["api_url"])
        layout.addRow("API URL:", self.api_url_edit)
        
        # API Password/Key
        self.api_password_edit = QLineEdit(self.settings["api_password"])
        layout.addRow("API Key:", self.api_password_edit)
        
        # Default language
        self.language_edit = QLineEdit(self.settings["language"])
        layout.addRow("Default Language:", self.language_edit)
        
        # Max chunk size
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(256, 8192)
        self.chunk_size_spin.setSingleStep(128)
        self.chunk_size_spin.setValue(self.settings["max_chunk_size"])
        layout.addRow("Max Chunk Size:", self.chunk_size_spin)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                          QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)
    
    def get_settings(self):
        """ Return the current settings """
        return {
            "api_url": self.api_url_edit.text(),
            "api_password": self.api_password_edit.text(),
            "language": self.language_edit.text(),
            "max_chunk_size": self.chunk_size_spin.value()
        }


class ChunkifyGUI(QMainWindow):
    """ Main application window for Chunkify GUI """
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings
        self.settings = {
            "api_url": "http://localhost:5001",
            "api_password": "",
            "language": "English",
            "max_chunk_size": 4096
        }
        
        self.input_files = []
        self.selected_task = "summary"
        
        # Main window setup
        self.setWindowTitle("Chunkify Text Processor")
        self.resize(800, 600)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Input file selection area
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No files selected")
        file_layout.addWidget(self.file_label)
        
        self.choose_file_btn = QPushButton("Choose Files")
        self.choose_file_btn.clicked.connect(self.select_input_files)
        file_layout.addWidget(self.choose_file_btn)
        
        main_layout.addLayout(file_layout)
        
        # Task selection group
        task_group = QGroupBox("Task")
        task_layout = QVBoxLayout(task_group)
        
        self.task_buttons = {}
        for task in ["summary", "translate", "correct", "distill"]:
            self.task_buttons[task] = QRadioButton(task.capitalize())
            self.task_buttons[task].clicked.connect(self.update_selected_task)
            task_layout.addWidget(self.task_buttons[task])
        
        # Set default task
        self.task_buttons["summary"].setChecked(True)
        
        main_layout.addWidget(task_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.config_btn = QPushButton("Configuration")
        self.config_btn.clicked.connect(self.open_config_dialog)
        controls_layout.addWidget(self.config_btn)
        
        self.process_btn = QPushButton("Process Files")
        self.process_btn.clicked.connect(self.process_files)
        self.process_btn.setEnabled(False)  # Disabled until files are selected
        controls_layout.addWidget(self.process_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Output window
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        
        # Create a scroll area for the output
        scroll_area = QScrollArea()
        scroll_area.setWidget(output_group)
        scroll_area.setWidgetResizable(True)
        
        main_layout.addWidget(scroll_area)
        
        # Setup output redirection
        self.redirector = OutputRedirector()
        self.redirector.text_output.connect(self.update_output)
        sys.stdout = self.redirector
    
    def update_selected_task(self):
        """ Update the selected task based on radio button selection """
        for task, button in self.task_buttons.items():
            if button.isChecked():
                self.selected_task = task
                break
    
    def select_input_files(self):
        """ Open file dialog to select input files """
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Input Files",
            "",
            "All Files (*.*)"
        )
        
        if files:
            self.input_files = files
            if len(files) == 1:
                self.file_label.setText(Path(files[0]).name)
            else:
                self.file_label.setText(f"{len(files)} files selected")
            
            self.process_btn.setEnabled(True)
    
    def open_config_dialog(self):
        """ Open the configuration dialog """
        dialog = ConfigDialog(self, self.settings)
        if dialog.exec():
            self.settings = dialog.get_settings()
    
    @pyqtSlot(str)
    def update_output(self, text):
        """ Update the output text widget """
        self.output_text.append(text)
        # Scroll to the bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output_text.setTextCursor(cursor)
    
    def process_files(self):
        """ Start processing the selected files """
        if not self.input_files:
            QMessageBox.warning(self, "Warning", "No input files selected")
            return
        
        # Clear output
        self.output_text.clear()
        
        # Disable the process button during processing
        self.process_btn.setEnabled(False)
        self.choose_file_btn.setEnabled(False)
        self.config_btn.setEnabled(False)
        
        # Set up progress tracking
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Process each file
        self.current_file_index = 0
        self.process_next_file()
    
    def process_next_file(self):
        """ Process the next file in the queue """
        if self.current_file_index >= len(self.input_files):
            # All files processed
            self.process_btn.setEnabled(True)
            self.choose_file_btn.setEnabled(True)
            self.config_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        
        input_path = self.input_files[self.current_file_index]
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_{self.selected_task}.txt"
        
        # Update progress
        progress_value = int((self.current_file_index / len(self.input_files)) * 100)
        self.progress_bar.setValue(progress_value)
        
        # Add file header to output
        self.update_output(f"\n\n--- Processing {Path(input_path).name} ({self.current_file_index + 1}/{len(self.input_files)}) ---\n")
        
        # Start processing in a separate thread
        self.worker = WorkerThread(
            api_url=self.settings["api_url"],
            input_path=input_path,
            task=self.selected_task,
            output_path=output_path,
            language=self.settings["language"],
            max_chunk_size=self.settings["max_chunk_size"],
            api_password=self.settings["api_password"]
        )
        
        self.worker.finished.connect(self.on_file_processed)
        self.worker.start()
    
    def on_file_processed(self, exit_code):
        """ Handle completion of file processing """
        if exit_code != 0:
            self.update_output(f"\nError processing file {Path(self.input_files[self.current_file_index]).name}")
        
        # Move to next file
        self.current_file_index += 1
        self.process_next_file()
    
    def closeEvent(self, event):
        """ Handle window close event """
        # Restore stdout
        sys.stdout = sys.__stdout__
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChunkifyGUI()
    window.show()
    sys.exit(app.exec())
