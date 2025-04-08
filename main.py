import sys
import os
import time
import whisper
import markdown
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QProgressBar, 
                           QTabWidget, QTextEdit, QCheckBox, QRadioButton, QGroupBox, 
                           QLineEdit, QMessageBox, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDir
from PyQt6.QtGui import QTextDocument

class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, input_file, model_size, use_gpu):
        super().__init__()
        self.input_file = input_file
        self.model_size = model_size
        self.use_gpu = use_gpu
        
    def run(self):
        try:
            # Emit starting progress
            self.progress_signal.emit(10)
            
            # Load model based on selected size
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self.progress_signal.emit(25)
            
            # Load the model
            model = whisper.load_model(self.model_size, device=device)
            self.progress_signal.emit(50)
            
            # Transcribe audio
            result = model.transcribe(self.input_file)
            self.progress_signal.emit(90)
            
            # Get transcription text
            text = result['text']
            self.progress_signal.emit(100)
            
            # Emit the transcribed text
            self.finished_signal.emit(text)
            
        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Whisper Audio Transcription")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.input_file = None
        self.transcribed_text = ""
        self.default_save_dir = os.path.join(os.getcwd(), "transcribed_text")
        
        # Ensure directories exist
        os.makedirs(os.path.join(os.getcwd(), "uploaded_audio"), exist_ok=True)
        os.makedirs(self.default_save_dir, exist_ok=True)
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # File selection section
        file_group = QGroupBox("Audio File")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_label = QLabel("No file selected")
        select_file_button = QPushButton("Select Audio File")
        select_file_button.clicked.connect(self.select_audio_file)
        
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(select_file_button)
        
        # Model selection section
        model_group = QGroupBox("Transcription Settings")
        model_layout = QVBoxLayout(model_group)
        
        model_size_layout = QHBoxLayout()
        model_size_layout.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("medium")
        model_size_layout.addWidget(self.model_combo)
        
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")
        
        # Default to CPU, enable GPU if available
        self.cpu_radio.setChecked(True)
        self.gpu_radio.setEnabled(torch.cuda.is_available())
        if not torch.cuda.is_available():
            self.gpu_radio.setToolTip("CUDA not available on this system")
        
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        
        model_layout.addLayout(model_size_layout)
        model_layout.addLayout(device_layout)
        
        # Output settings section
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(QLabel("Save Directory:"))
        self.save_dir_edit = QLineEdit(self.default_save_dir)
        self.save_dir_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.select_save_directory)
        
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(browse_button)
        
        output_layout.addLayout(save_dir_layout)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Transcribe button
        self.transcribe_button = QPushButton("Transcribe Audio")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Tab widget for raw text and markdown
        self.tabs = QTabWidget()
        
        # Raw text tab
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setReadOnly(True)
        self.tabs.addTab(self.raw_text_edit, "Raw Transcript")
        
        # Markdown tab
        markdown_widget = QWidget()
        markdown_layout = QVBoxLayout(markdown_widget)
        
        self.markdown_edit = QTextEdit()
        self.markdown_preview = QTextEdit()
        self.markdown_preview.setReadOnly(True)
        
        # Connect markdown edit to preview
        self.markdown_edit.textChanged.connect(self.update_markdown_preview)
        
        markdown_splitter = QSplitter(Qt.Orientation.Horizontal)
        markdown_splitter.addWidget(self.markdown_edit)
        markdown_splitter.addWidget(self.markdown_preview)
        
        markdown_layout.addWidget(markdown_splitter)
        self.tabs.addTab(markdown_widget, "Markdown")
        
        results_layout.addWidget(self.tabs)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.prettify_button = QPushButton("Prettify to Markdown")
        self.prettify_button.clicked.connect(self.prettify_to_markdown)
        self.prettify_button.setEnabled(False)
        
        self.export_raw_button = QPushButton("Export Raw Text")
        self.export_raw_button.clicked.connect(lambda: self.export_text(False))
        self.export_raw_button.setEnabled(False)
        
        self.export_markdown_button = QPushButton("Export Markdown")
        self.export_markdown_button.clicked.connect(lambda: self.export_text(True))
        self.export_markdown_button.setEnabled(False)
        
        export_layout.addWidget(self.prettify_button)
        export_layout.addWidget(self.export_raw_button)
        export_layout.addWidget(self.export_markdown_button)
        
        # Add all components to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(self.transcribe_button)
        main_layout.addWidget(results_group)
        main_layout.addLayout(export_layout)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
    
    def select_audio_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4);;All Files (*)"
        )
        
        if file_path:
            self.input_file = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.transcribe_button.setEnabled(True)
    
    def select_save_directory(self):
        dir_dialog = QFileDialog()
        dir_path = dir_dialog.getExistingDirectory(
            self, "Select Save Directory", self.default_save_dir
        )
        
        if dir_path:
            self.save_dir_edit.setText(dir_path)
    
    def start_transcription(self):
        if not self.input_file:
            QMessageBox.warning(self, "Error", "Please select an audio file first.")
            return
        
        # Disable buttons during transcription
        self.transcribe_button.setEnabled(False)
        self.prettify_button.setEnabled(False)
        self.export_raw_button.setEnabled(False)
        self.export_markdown_button.setEnabled(False)
        
        # Update status
        self.status_label.setText("Transcribing...")
        self.progress_bar.setValue(0)
        
        # Get selected model and device
        model_size = self.model_combo.currentText()
        use_gpu = self.gpu_radio.isChecked()
        
        # Create and start the transcription thread
        self.thread = TranscriptionThread(self.input_file, model_size, use_gpu)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.transcription_finished)
        self.thread.error_signal.connect(self.transcription_error)
        self.thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def transcription_finished(self, text):
        # Update the raw text
        self.transcribed_text = text
        self.raw_text_edit.setText(text)
        
        # Update status and enable buttons
        self.status_label.setText("Transcription complete")
        self.transcribe_button.setEnabled(True)
        self.prettify_button.setEnabled(True)
        self.export_raw_button.setEnabled(True)
        
        # Switch to the raw text tab
        self.tabs.setCurrentIndex(0)
    
    def transcription_error(self, error_message):
        # Display error message
        QMessageBox.critical(self, "Error", f"Transcription failed: {error_message}")
        
        # Update status and re-enable transcribe button
        self.status_label.setText("Error during transcription")
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def prettify_to_markdown(self):
        # Simple conversion to markdown - this can be enhanced
        raw_text = self.transcribed_text
        
        # Basic markdown formatting - paragraphs
        md_text = "\n\n".join(p.strip() for p in raw_text.split("\n\n"))
        
        # Set the markdown text
        self.markdown_edit.setText(md_text)
        
        # Switch to markdown tab
        self.tabs.setCurrentIndex(1)
        
        # Enable export markdown button
        self.export_markdown_button.setEnabled(True)
    
    def update_markdown_preview(self):
        # Convert markdown to HTML and display in preview
        md_text = self.markdown_edit.toPlainText()
        html = markdown.markdown(md_text)
        
        # Set HTML in preview
        self.markdown_preview.setHtml(html)
    
    def export_text(self, use_markdown=False):
        # Determine file path
        if self.input_file:
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            save_dir = self.save_dir_edit.text()
            
            # Create file extension and content based on export type
            if use_markdown:
                extension = ".md"
                content = self.markdown_edit.toPlainText()
            else:
                extension = ".txt"
                content = self.raw_text_edit.toPlainText()
            
            # Create default filename
            default_filename = f"{base_name}{extension}"
            save_path = os.path.join(save_dir, default_filename)
            
            # Get save file path from dialog
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "Save Transcription", save_path,
                f"Text Files (*{extension});;All Files (*)"
            )
            
            # Save file if path was selected
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    QMessageBox.information(
                        self, "Success", f"Transcription saved to {file_path}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error", f"Failed to save file: {str(e)}"
                    )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 