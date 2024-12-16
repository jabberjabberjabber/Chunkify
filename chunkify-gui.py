import sys
import os
import json
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QRadioButton, QButtonGroup,
                           QFileDialog, QListWidget, QLabel, QTextEdit, QLineEdit,
                           QGroupBox, QPlainTextEdit, QDialog, QComboBox, QSpinBox,
                           QDoubleSpinBox, QMenuBar, QMenu, QAction, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from chunkify import LLMConfig, LLMProcessor, check_api

##
## GUI written mostly by Claude Sonnet 3.5
##

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)
    
    def __init__(self, config, task, instruction, files, selected_template=None):
        super().__init__()
        self.config = config
        self.task = task
        self.instruction = instruction
        self.files = files
        self.selected_template = selected_template
    def run(self):
        try:
            self.processor = LLMProcessor(self.config, self.task)
            
            # Override the monitor function for the GUI
            def gui_monitor():
                generating = False
                last_result = ""
                payload = {'genkey': self.processor.genkey}
                while not self.processor.generated:
                    result = self.processor._call_api("check", payload)
                    if not result:
                        time.sleep(2)
                        continue
                    if result != last_result:  # Only update if the text has changed
                        last_result = result
                        # Send a clear signal before new text
                        self.progress_signal.emit("<<CLEAR>>")
                        self.progress_signal.emit(f"{result}")
                    time.sleep(1)    

            # Replace the monitor function
            self.processor._monitor_generation = gui_monitor
            
            results = []
            
            for file_path in self.files:
                self.progress_signal.emit(f"Processing {file_path}...")
                content, metadata = self.processor._get_content(file_path)
                
                if self.task == "custom":
                    responses = self.processor.process_in_chunks(self.instruction, content)
                else:
                    responses = self.processor.route_task(self.task, content)
                
                # Create output filename
                path = Path(file_path)
                output_path = path.parent / f"{path.stem}_processed{path.suffix}"
                
                # Write output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"File: {metadata.get('resourceName', 'Unknown')}\n")
                    f.write(f"Type: {metadata.get('Content-Type', 'Unknown')}\n")
                    f.write(f"Encoding: {metadata.get('Content-Encoding', 'Unknown')}\n")
                    f.write(f"Length: {metadata.get('Content-Length', 'Unknown')}\n\n")
                    for response in responses:
                        f.write(f"{response}\n\n")
                
                results.append((file_path, output_path))
                
            self.finished_signal.emit(results)
            
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")
        
class ChunkerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_file = './chunkify_config.json'
        self.config = self.load_config()
        self.selected_template = None
        self.initUI()
        self.api_ready = False
        
        # Start API check timer
        from PyQt5.QtCore import QTimer
        self.api_timer = QTimer()
        self.api_timer.timeout.connect(self.check_api)
        self.api_timer.start(2000)  # Check every 2 seconds
        
        #self.check_api()
    def check_api(self):
        try:
            #while not self.api_ready: 
            result = check_api(self.config.api_url)    
            if result:
                self.api_ready = True
                self.process_button.setEnabled(True)
                self.output_text.appendPlainText("API is ready - you can now process files.")
                self.api_timer.stop()
            else:
                if not self.api_ready:  # Only show loading message if not yet ready
                    self.process_button.setEnabled(False)
                    self.output_text.setPlainText("Waiting for API to become available...")
            #time.sleep(2)
        except Exception as e:
            if not self.api_ready:
                self.process_button.setEnabled(False)
                self.output_text.setPlainText(f"Waiting for API...\n")        
                
    def initUI(self):
        self.setWindowTitle('Text Processing GUI')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # File selection area
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        file_buttons = QHBoxLayout()
        
        add_button = QPushButton('Add Files')
        add_button.clicked.connect(self.add_files)
        remove_button = QPushButton('Remove Selected')
        remove_button.clicked.connect(self.remove_files)
        
        file_buttons.addWidget(add_button)
        file_buttons.addWidget(remove_button)
        
        file_layout.addLayout(file_buttons)
        file_layout.addWidget(self.file_list)
        file_group.setLayout(file_layout)
        self.file_list.setFixedHeight(100)  # Adjust this value as needed
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.file_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # For the output text, we can use size policies to make it expand

        # Task selection area
        task_group = QGroupBox("Task Selection")
        task_layout = QVBoxLayout()
        
        self.task_group = QButtonGroup()
        tasks = ['summary', 'translate', 'distill', 'correct']
        
        for i, task in enumerate(tasks):
            radio = QRadioButton(task.capitalize())
            self.task_group.addButton(radio, i)
            task_layout.addWidget(radio)
            if task == 'summary':
                radio.setChecked(True)
       
        
        task_group.setLayout(task_layout)
        
        # Output area
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        self.output_text = QPlainTextEdit()
        self.output_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: black;
                color: white;
                font-family: Consolas, Monaco, monospace;
            }
        """)
        self.output_text.setReadOnly(True)
        self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)

        # Process button
        self.process_button = QPushButton('Process Files')
        self.process_button.clicked.connect(self.process_files)
        
        # Add everything to main layout
        layout.addWidget(file_group)
        layout.addWidget(task_group)
        layout.addWidget(output_group)
        layout.addWidget(self.process_button)
        
        main_widget.setLayout(layout)
        file_group.setMaximumHeight(200)  # Adjust this value as needed

        # Keep task group compact 
        task_group.setMaximumHeight(150)  # Adjust this value as needed

        # Make output group expand
        output_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Keep process button from expanding
        self.process_button.setFixedHeight(30)  # Optional, for consistent button height
        # Menu bar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        
        config_action = QAction('Configuration', self)
        config_action.triggered.connect(self.show_config_dialog)
        settings_menu.addAction(config_action)
        
        # Initialize processing thread as None
        self.processing_thread = None

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files to process",
            "",
            "All Files (*.*)"
        )
        for file in files:
            self.file_list.addItem(file)
            
    def remove_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
    def process_files(self):
        if self.processing_thread and self.processing_thread.isRunning():
            return
                
        files = [self.file_list.item(i).text() 
                 for i in range(self.file_list.count())]
        
        if not files:
            self.output_text.appendPlainText("Error: No files selected")
            return
                
        # Clear the output box before starting
        self.output_text.clear()
                
        # Get selected task
        task_id = self.task_group.checkedId()
        tasks = ['summary', 'translate', 'distill', 'correct']
        task = tasks[task_id]
        instruction = ""
        
        config = self.config
        
        self.processing_thread = ProcessingThread(
            config=config, 
            task=task, 
            instruction=instruction, 
            files=files,
            selected_template=self.selected_template
        )
        
        # Update progress handler to check for clear signal
        def handle_progress(msg):
            if msg == "<<CLEAR>>":
                self.output_text.clear()
            else:
                self.output_text.appendPlainText(msg)
                
        self.processing_thread.progress_signal.connect(handle_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()
        
        self.process_button.setEnabled(False)

        
    def processing_finished(self, results):
        self.output_text.appendPlainText("\nProcessing completed!")
        for input_file, output_file in results:
            self.output_text.appendPlainText(
                f"\nProcessed {input_file}\nOutput saved to {output_file}"
            )
        self.process_button.setEnabled(True)

    def show_config_dialog(self):
        dialog = ConfigDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            # Existing config updates...
            self.config.api_url = dialog.api_url_input.text()
            self.config.api_password = dialog.api_password_input.text()
            self.config.temp = dialog.temp_input.value()
            self.config.rep_pen = dialog.rep_pen_input.value()
            self.config.top_k = dialog.top_k_input.value()
            self.config.top_p = dialog.top_p_input.value()
            self.config.min_p = dialog.min_p_input.value()
            self.selected_template = dialog.template_combo.currentText()
            
            # Update translation language in config
            self.config.translation_language = dialog.translation_language_input.text()
            
            # Save config to file
            self.save_config()
            
            # Immediately check API with new settings
            self.api_ready = False
            self.check_api()
            
            # If we have an active processing thread, update its config and refresh instructions
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.config = self.config
                if hasattr(self.processing_thread, 'processor'):
                    self.processing_thread.processor.update_config(self.config)
            
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file) as f:
                    config_data = json.load(f)
                    
                
                # Load template if it exists
                self.selected_template = config_data.pop('selected_template', None)
                
                return LLMConfig(**config_data)
            else:
                # Return default config
                return LLMConfig(
                    templates_directory="./templates",
                    api_url="http://localhost:5001",
                    api_password=""
                )
        except Exception as e:
            print(f"Error loading config: {e}")
            return LLMConfig(
                templates_directory="./templates",
                api_url="http://localhost:5001",
                api_password=""
            )
            
    def save_config(self):
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'templates_directory': self.config.templates_directory,
                'api_url': self.config.api_url,
                'api_password': self.config.api_password,
                'temp': self.config.temp,
                'rep_pen': self.config.rep_pen,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'min_p': self.config.min_p,
                'selected_template': self.selected_template,
                'translation_language': self.config.translation_language
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        
        except Exception as e:
            print(f"Error saving config: {e}")
            
            
class ConfigDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Configuration')
        self.setModal(True)
        layout = QVBoxLayout()
        
        # API Settings
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        
        self.api_url_input = QLineEdit(self.config.api_url)
        api_layout.addWidget(QLabel("API URL:"))
        api_layout.addWidget(self.api_url_input)
        
        self.api_password_input = QLineEdit(self.config.api_password)
        self.api_password_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(QLabel("API Password:"))
        api_layout.addWidget(self.api_password_input)
        
        self.translation_language_input = QLineEdit(self.config.translation_language)
        #self.translation_language_input.setEchoMode(QLineEdit.translation_language)
        api_layout.addWidget(QLabel("Translate to language:"))
        api_layout.addWidget(self.translation_language_input)
        
        api_group.setLayout(api_layout)
        
        # Sampler Settings
        sampler_group = QGroupBox("Sampler Settings")
        sampler_layout = QVBoxLayout()
        
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(0, 2)
        self.temp_input.setSingleStep(0.1)
        self.temp_input.setValue(self.config.temp)
        sampler_layout.addWidget(QLabel("Temperature:"))
        sampler_layout.addWidget(self.temp_input)
        
        self.rep_pen_input = QDoubleSpinBox()
        self.rep_pen_input.setRange(0, 10)
        self.rep_pen_input.setSingleStep(0.1)
        self.rep_pen_input.setValue(self.config.rep_pen)
        sampler_layout.addWidget(QLabel("Repetition Penalty:"))
        sampler_layout.addWidget(self.rep_pen_input)
        
        self.top_k_input = QSpinBox()
        self.top_k_input.setRange(0, 100)
        self.top_k_input.setValue(self.config.top_k)
        sampler_layout.addWidget(QLabel("Top K:"))
        sampler_layout.addWidget(self.top_k_input)
        
        self.top_p_input = QDoubleSpinBox()
        self.top_p_input.setRange(0, 1)
        self.top_p_input.setSingleStep(0.1)
        self.top_p_input.setValue(self.config.top_p)
        sampler_layout.addWidget(QLabel("Top P:"))
        sampler_layout.addWidget(self.top_p_input)
        
        self.min_p_input = QDoubleSpinBox()
        self.min_p_input.setRange(0, 1)
        self.min_p_input.setSingleStep(0.01)
        self.min_p_input.setValue(self.config.min_p)
        sampler_layout.addWidget(QLabel("Min P:"))
        sampler_layout.addWidget(self.min_p_input)
        
        sampler_group.setLayout(sampler_layout)
        
        
        # Template Selection
        template_group = QGroupBox("Template Selection")
        template_layout = QVBoxLayout()
        
        self.template_combo = QComboBox()
        # Load available templates
        template_path = Path(self.config.templates_directory)
        if template_path.exists():
            templates = [f.stem for f in template_path.glob('*.json')]
            self.template_combo.addItems(['Auto'] + templates)
        else:
            self.template_combo.addItem('Auto')
        
        template_layout.addWidget(QLabel("Model Template:"))
        template_layout.addWidget(self.template_combo)
        
        template_group.setLayout(template_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        # Add all groups to main layout
        layout.addWidget(api_group)
        layout.addWidget(sampler_group)
        layout.addWidget(template_group)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

def main():
    app = QApplication(sys.argv)
    gui = ChunkerGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
