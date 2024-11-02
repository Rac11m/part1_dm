import sys
import pandas as pd
import numpy as np
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTableView, QLabel, QMessageBox, QFileDialog, QDialog, QFormLayout, QLineEdit, QComboBox
from PyQt5.QtCore import QAbstractTableModel, Qt
import matplotlib.pyplot as plt

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return str(self._df.iloc[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        elif orientation == Qt.Vertical:
            return str(self._df.index[section])

    def update_data(self, new_df):
        self._df = new_df
        self.layoutChanged.emit()  # Notify that the layout has changed

class DataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.figure = None 
        self.df = pd.DataFrame()

        # Setup main window
        self.setWindowTitle('Data Analysis Tool')
        self.setGeometry(100, 100, 1200, 800)

        # Create a menu bar
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        import_action = QAction('Import', self)
        visualize_action = QAction('Visualize', self)
        save_action = QAction('Save', self)
        file_menu.addAction(import_action)
        file_menu.addAction(visualize_action)
        file_menu.addAction(save_action)

        # Data Manipulation menu
        data_menu = menubar.addMenu('Data Manipulation')
        update_action = QAction('Update/Delete Instance', self)
        data_menu.addAction(update_action)

        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        stats_action = QAction('Statistics', self)
        boxplot_action = QAction('Boxplots', self)
        histogram_action = QAction('Histograms', self)
        scatter_action = QAction('Scatter Plots', self)
        analysis_menu.addAction(stats_action)
        analysis_menu.addAction(boxplot_action)
        analysis_menu.addAction(histogram_action)
        analysis_menu.addAction(scatter_action)

        # Central widget to display content based on menu actions
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create a horizontal layout for centering the label
        center_layout = QHBoxLayout()
        self.placeholder_label = QLabel("Select an option from the menu to get started.")
        center_layout.addStretch()  # Add stretchable space before the label
        center_layout.addWidget(self.placeholder_label)
        center_layout.addStretch()  # Add stretchable space after the label

        # Add the centered layout to the main layout
        self.layout.addLayout(center_layout)

        # Connect menu actions to functions
        import_action.triggered.connect(self.display_import_view)
        visualize_action.triggered.connect(self.visualize_data)
        # stats_action.triggered.connect(self.display_stats_view)
        save_action.triggered.connect(self.save_data)
        update_action.triggered.connect(self.open_update_delete_dialog)
        
        stats_action.triggered.connect(self.analyze_data)
        boxplot_action.triggered.connect(self.construct_boxplot)
        histogram_action.triggered.connect(self.construct_histogram)
        scatter_action.triggered.connect(self.create_scatter_plot)

    def analyze_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
        dialog = AnalysisDialog(self.df, self)
        dialog.exec_()

    def construct_boxplot(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
        # Similar to analyze_data but for boxplots

    def construct_histogram(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
        # Similar to analyze_data but for histograms

    def create_scatter_plot(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
    def clear_layout(self):
        # Clear existing widgets in the layout
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_import_view(self):
        self.clear_layout()
        import_button = QPushButton('Import Data')
        import_button.clicked.connect(self.import_data)
        self.layout.addWidget(import_button)
        self.data_table = QTableView()
        self.layout.addWidget(self.data_table)

    def display_stats_view(self):
        self.clear_layout()
        stats_label = QLabel("Statistics and analysis of your dataset.")
        self.layout.addWidget(stats_label)

    def import_data(self):
        # Open a file dialog to select a CSV file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_name:  # Check if a file was selected
            try:
                # Load the CSV file into a pandas DataFrame
                self.df = pd.read_csv(file_name)

                # Display the DataFrame in the table view
                model = PandasModel(self.df)  # Create a model from the DataFrame
                self.data_table.setModel(model)  # Set the model for the QTableView
            except Exception as e:
                # Show an error message if loading fails
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def visualize_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return

        # Close the previous figure if it exists
        if self.figure:
            plt.close(self.figure)

        # Create a new figure
        self.figure = plt.figure(figsize=(10, 6))
        self.df.hist(bins=30, figsize=(10, 10), grid=False)
        plt.tight_layout()
        plt.show()

    def save_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data available to save.")
            return

        # Open a file dialog to select the save location and filename
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_name:  # Check if a file was selected
            try:
                self.df.to_csv(file_name, index=False)  # Save the DataFrame to a CSV file without the index
                QMessageBox.information(self, "Success", f"Data saved successfully to {file_name}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def open_update_delete_dialog(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return

        dialog = UpdateDeleteDialog(self.df, self.update_dataframe, self)
        dialog.exec_()

    def update_dataframe(self, new_df):
        self.df = new_df  # Update the main DataFrame
        model = PandasModel(self.df)  # Create a new model with the updated DataFrame
        self.data_table.setModel(model)  # Set the new model for the QTableView



class AnalysisDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("Statistical Analysis")
        self.setGeometry(100, 100, 400, 300)
        self.layout = QVBoxLayout(self)

        # Analysis type selection
        self.analysis_type_selector = QComboBox()
        self.analysis_type_selector.addItems(["Central Tendency", "Dispersion Measures", "Missing & Unique Values", "Boxplots", "Histograms", "Scatter Plot"])
        self.layout.addWidget(QLabel("Select type of analysis:"))
        self.layout.addWidget(self.analysis_type_selector)

        # Column selector for analysis
        self.column_selector = QComboBox()
        self.column_selector.addItems(df.columns)
        self.layout.addWidget(QLabel("Select column for analysis:"))
        self.layout.addWidget(self.column_selector)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_analysis)
        self.layout.addWidget(self.analyze_button)

        self.result_label = QLabel()
        self.layout.addWidget(self.result_label)

    def perform_analysis(self):
        column = self.column_selector.currentText()
        analysis_type = self.analysis_type_selector.currentText()

        if analysis_type == "Central Tendency":
                tendencies = self.calculate_central_tendency(column)
                self.result_label.setText(tendencies)
 
        elif analysis_type == "Dispersion Measures":
            if self.df[column].dtype in [np.float64, np.int64]:
                dispersion = self.calculate_dispersion(column)
                self.result_label.setText(dispersion)
            else:
                QMessageBox.warning(self, "Warning", "Selected column is not numeric.")

        elif analysis_type == "Missing & Unique Values":
            missing_unique = self.calculate_missing_unique(column)
            self.result_label.setText(missing_unique)

        elif analysis_type == "Boxplots":
            self.construct_boxplot(column)

        elif analysis_type == "Histograms":
            self.construct_histogram(column)

        elif analysis_type == "Scatter Plot":
            self.select_scatter_plot_columns()

    def calculate_central_tendency(self, column):
        if pd.api.types.is_numeric_dtype(self.df[column]):
            mean = self.df[column].mean()
            median = self.df[column].median()
            mode_series = self.df[column].mode()
            mode = mode_series[0] if not mode_series.empty else np.nan

            if np.isclose(mean, median) and np.isclose(mean, mode):
                symmetry = "Symmetrical"
            elif mean > median > mode:
                symmetry = "Right-Skewed"
            elif mean < median < mode:
                symmetry = "Left-Skewed"
            else:
                symmetry = "Asymmetrical"                
            
            return f"{column}: Mean = {mean:.3f}, Median = {median:.3f}, Mode = {mode:.3f} | Symmetry: {symmetry}"
        else:
            mode_series = self.df[column].mode()
            mode = mode_series[0] if not mode_series.empty else np.nan
            return f"{column}: Only mode calculated (non-numeric attribut), Mode: {mode}"   
        
    def calculate_dispersion(self, column):
        range_value = self.df[column].max() - self.df[column].min()
        variance = self.df[column].var()
        std_dev = self.df[column].std()
        q1 = self.df[column].quantile(q=0.25)
        q3 = self.df[column].quantile(q=0.75)
        iqr = q3 - q1
        iqr_1_5 = 1.5 * iqr
        
        lower_bound = q1 - iqr_1_5
        upper_bound = q3 + iqr_1_5
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column]
        num_outliers = outliers.count()

        return (f"Minimum: {self.df[column].min()}, Maximum: {self.df[column].max()}, Range: {range_value:.3f}, Variance: {variance:.3f}, Standard Deviation: {std_dev:.3f},\nQ1: {q1:.3f}, Q3: {q3:.3f}, IQR: {iqr:.3f}, Lower Bound: {lower_bound:.3f}, Upper Bound: {upper_bound:.3f}, Number of Outliers: {num_outliers}")

    def calculate_missing_unique(self, column):
        total_count = self.df[column].shape[0]
        missing_count = self.df[column].isnull().sum()
        unique_count = self.df[column].nunique()
        missing_percentage = (missing_count / total_count) * 100
        return (f"Missing values count: {missing_count}, Missing values percentage: {missing_percentage:.2f}%, Unique values count: {unique_count}")

    def construct_boxplot(self, column):
        plt.figure(figsize=(10, 8))
        plt.boxplot(self.df[column].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
        
        q1 = self.df[column].quantile(q=0.25)
        q3 = self.df[column].quantile(q=0.75)
        iqr = q3 - q1
        iqr_1_5 = 1.5 * iqr
        
        lower_bound = q1 - iqr_1_5
        upper_bound = q3 + iqr_1_5
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column]

        for i, outlier_value in outliers.items():
            plt.text(outlier_value, 1, f'{outlier_value:.2f}', color='red', fontsize=9, ha='left')
    

        plt.title(f'Boxplot for {column} (With outliers)')
        plt.xlabel('Values')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    def construct_histogram(self, column):
        plt.figure(figsize=(10, 8))
        plt.hist(self.df[column].dropna(), bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        
        
        if pd.api.types.is_numeric_dtype(self.df[column]):
            mean = self.df[column].mean()
            median = self.df[column].median()
        
            plt.axvline(mean, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean:.2f}')
            plt.axvline(median, color='green', linestyle='-', linewidth=1, label=f'Median: {median:.2f}')

        
        plt.title(f'Histogram for {column}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    def select_scatter_plot_columns(self):
        # Open a new dialog or modify the existing dialog to select two columns for the scatter plot
        scatter_dialog = ScatterPlotDialog(self.df, self)
        scatter_dialog.exec_()


class ScatterPlotDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("Select Columns for Scatter Plot")
        self.setGeometry(100, 100, 400, 200)
        self.layout = QVBoxLayout(self)

        self.column_x_selector = QComboBox()
        self.column_x_selector.addItems(df.columns)
        self.layout.addWidget(QLabel("Select X-axis column:"))
        self.layout.addWidget(self.column_x_selector)

        self.column_y_selector = QComboBox()
        self.column_y_selector.addItems(df.columns)
        self.layout.addWidget(QLabel("Select Y-axis column:"))
        self.layout.addWidget(self.column_y_selector)

        self.scatter_button = QPushButton("Create Scatter Plot")
        self.scatter_button.clicked.connect(self.create_scatter_plot)
        self.layout.addWidget(self.scatter_button)




    def create_scatter_plot(self):
        column_x = self.column_x_selector.currentText()
        column_y = self.column_y_selector.currentText()
        scatter_plot(self.df, column_x, column_y)  # Assuming scatter_plot function is defined globally
        self.close()

def scatter_plot(df,column_x, column_y):
    data_to_plot = df[[column_x, column_y]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(df[column_x], df[column_y], alpha=0.6)

    correlation = df[column_x].corr(df[column_y])


    plt.title(f'Scatter Plot: {column_x} vs {column_y}\nCorrelation: {correlation:.2f}')
    plt.xlabel(column_x)
    plt.ylabel(column_y)

    sns.regplot(x=column_x, y=column_y, data=df, scatter=False, color='red', line_kws={"linewidth": 1})

    plt.grid()
    plt.show() 
    


class UpdateDeleteDialog(QDialog):
    def __init__(self, df, update_callback, parent=None):
        super().__init__(parent)
        self.df = df
        self.update_callback = update_callback
        self.setWindowTitle("Update/Delete Instance")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        # Dropdown for selecting rows
        self.row_selector = QComboBox()
        self.update_row_selector()  # Fill the combo box with the initial DataFrame rows
        self.layout.addWidget(self.row_selector)

        # Dropdown for selecting columns
        self.column_selector = QComboBox()
        self.column_selector.addItems(self.df.columns)  # Populate with DataFrame columns
        self.layout.addWidget(self.column_selector)

        # Input field for updating
        self.update_field = QLineEdit()
        self.update_field.setPlaceholderText("Enter new value")
        self.layout.addWidget(self.update_field)

        # Buttons for update and delete
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_instance)
        self.layout.addWidget(self.update_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_instance)
        self.layout.addWidget(self.delete_button)

        self.setLayout(self.layout)

    def update_row_selector(self):
        """Update the row selector options based on the current DataFrame."""
        self.row_selector.clear()
        self.row_selector.addItems([str(i) for i in range(len(self.df))])  # Add current row indices

    def update_instance(self):
        selected_row = self.row_selector.currentIndex()
        selected_column = self.column_selector.currentText()
        new_value = self.update_field.text()
        
        if not new_value:
            QMessageBox.warning(self, "Warning", "Please enter a value to update.")
            return

        # Update the selected column in the selected row
        column_index = self.df.columns.get_loc(selected_column)
        self.df.iat[selected_row, column_index] = new_value
        QMessageBox.information(self, "Success", f"Updated row {selected_row}, column '{selected_column}' with value: {new_value}")

        # Notify the main application to update the DataFrame
        self.update_callback(self.df)

    def delete_instance(self):
        selected_row = self.row_selector.currentIndex()
        self.df = self.df.drop(selected_row).reset_index(drop=True)  # Delete the selected row and reset the index
        QMessageBox.information(self, "Success", f"Deleted row {selected_row}.")

        # Notify the main application to update the DataFrame
        self.update_callback(self.df)
        self.update_row_selector()  # Refresh the row selector to reflect changes

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = DataApp()
    mainWin.show()
    sys.exit(app.exec_())
