import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QLabel, QPushButton, QScrollArea,
                             QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QColor
from PyQt6 import QtCore


class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.sequence = []
        self.history_line_edits = {}
        self.predictions = []
        self.model = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Aviator Prediction Model")
        self.setFixedSize(600, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # History area
        self.history_layout = QGridLayout()
        self.history_widget = QWidget()
        self.history_widget.setLayout(self.history_layout)
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidget(self.history_widget)
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFixedHeight(200)
        main_layout.addWidget(self.history_scroll)

        # Predictions area
        self.predictions_layout = QGridLayout()
        self.predictions_widget = QWidget()
        self.predictions_widget.setLayout(self.predictions_layout)
        self.predictions_scroll = QScrollArea()
        self.predictions_scroll.setWidget(self.predictions_widget)
        self.predictions_scroll.setWidgetResizable(True)
        self.predictions_scroll.setFixedHeight(100)
        main_layout.addWidget(self.predictions_scroll)

        # Input fields and labels
        input_layout = QVBoxLayout()

        predictions_layout = QHBoxLayout()
        predictions_label = QLabel("Number of Predictions:")
        predictions_layout.addWidget(predictions_label)
        self.predictions_entry = QLineEdit()
        self.predictions_entry.setFixedWidth(50)
        predictions_layout.addWidget(self.predictions_entry)
        input_layout.addLayout(predictions_layout)

        sequence_layout = QHBoxLayout()
        sequence_label = QLabel("Input Sequence:")
        sequence_layout.addWidget(sequence_label)
        self.sequence_entry = QLineEdit()
        self.sequence_entry.setFixedWidth(100)
        sequence_layout.addWidget(self.sequence_entry)
        input_layout.addLayout(sequence_layout)

        self.sequence_entry.returnPressed.connect(self.add_to_history)

        main_layout.addLayout(input_layout)

        # Predict button
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        main_layout.addWidget(self.predict_button)

        self.setLayout(main_layout)

    def add_to_history(self):
        try:
            value = float(self.sequence_entry.text())
            self.sequence.append(value)
            self.color_code(value, QColor(90, 30, 205), self.history_layout)  # Default purple color
            self.sequence_entry.clear()
            self.manage_history()
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Please enter a valid float.")

    def color_code(self, value, color, layout):
        line_edit = QLineEdit(f"{value:.4f}")
        line_edit.setStyleSheet(f"""
            background-color: {color.name()};
            border-radius: 10px;
            margin: 2px;
        """)
        line_edit.setFixedSize(70, 30)
        line_edit.setReadOnly(True) 
        line_edit.editingFinished.connect(lambda: self.update_history(line_edit))
        line_edit.installEventFilter(self)
        layout.addWidget(line_edit, layout.count() // 5, layout.count() % 5)

        # Store the QLineEdit and its index in the dictionary
        self.history_line_edits[line_edit] = len(self.sequence)

    def update_history(self, line_edit):
        try:
            new_value = float(line_edit.text())
            index = self.history_line_edits[line_edit] - 1
            if index < len(self.sequence):  # Check if index is valid
                if new_value != self.sequence[index]:
                    self.sequence[index] = new_value
                    line_edit.setStyleSheet("background-color: #5a4fff; border-radius: 10px; margin: 2px;")  # Change color for edited values
            else:
                QMessageBox.warning(self, "Index Error", "Invalid index detected.")
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Please enter a valid float.")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick and isinstance(obj, QLineEdit):
            self.edit_value(obj)
        return super().eventFilter(obj, event)

    def edit_value(self, line_edit):
        line_edit.setReadOnly(False)
        line_edit.setFocus()
        line_edit.selectAll()

    def mousePressEvent(self, event):
        for line_edit in self.history_line_edits.keys():
            line_edit.setReadOnly(True)
        return super().mousePressEvent(event)

    def manage_history(self):
        if len(self.sequence) > 20:
            self.sequence.pop(0)
            self.refresh_history()

    def refresh_history(self):
        # Clear and refresh the history display
        for i in reversed(range(self.history_layout.count())):
            widget_to_remove = self.history_layout.itemAt(i).widget()
            self.history_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        for value in self.sequence:
            self.color_code(value, QColor(90, 30, 205), self.history_layout)

    def train_model(self):
        df = pd.DataFrame(self.sequence, columns=["multiplier"])
        window_size = 3
        for lag in range(1, window_size + 1):
            df[f'lag_{lag}'] = df["multiplier"].shift(lag)
        df.dropna(inplace=True)
        features = df.drop("multiplier", axis=1)
        targets = df["multiplier"]
        X_train, _, y_train, _ = train_test_split(features, targets, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self):
        try:
            num_predictions = int(self.predictions_entry.text())
            if len(self.sequence) >= 3:
                self.train_model()
                current_features = np.array(self.sequence[-3:]).reshape(1, -1)
                predictions = []
                for _ in range(num_predictions):
                    prediction = self.model.predict(current_features)
                    predictions.append(round(float(prediction), 4))
                    current_features = np.append(current_features[:, 1:], prediction).reshape(1, -1)
                self.predictions = predictions
                self.display_predictions()
                self.animate_button()
            else:
                QMessageBox.warning(self, "Insufficient Data", "Please input at least 3 values for prediction.")
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Please enter a valid number of predictions.")


    def display_predictions(self):
        # Clear the predictions display
        for i in reversed(range(self.predictions_layout.count())):
            widget_to_remove = self.predictions_layout.itemAt(i).widget()
            self.predictions_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        # Display new predictions
        for value in self.predictions:
            self.color_code(value, QColor(0, 255, 0), self.predictions_layout)  # Green color

    def animate_button(self):
        animation = QPropertyAnimation(self.predict_button, b"geometry")
        original_rect = self.predict_button.geometry()
        animation.setDuration(500)
        animation.setStartValue(QRect(original_rect))
        animation.setEndValue(QRect(original_rect.x() - 5, original_rect.y() - 5, original_rect.width() + 10, original_rect.height() + 10))
        animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        animation.setLoopCount(1)
        animation.setDirection(QPropertyAnimation.Direction.Forward)
        animation.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec())


