import sys
import pandas as pd
import numpy as np
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTableView, QLabel, QMessageBox, QFileDialog, QDialog, QFormLayout, QLineEdit, QComboBox, QTableWidgetItem, QInputDialog
from PyQt5.QtCore import QAbstractTableModel, Qt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

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
        stats_all = QAction('All', self)
        analysis_menu.addAction(stats_action)
        analysis_menu.addAction(stats_all)
        # analysis_menu.addAction(boxplot_action)
        # analysis_menu.addAction(histogram_action)
        # analysis_menu.addAction(scatter_action)

        # Data Cleaning menu
        data_cleaning_menu = menubar.addMenu('Data Cleaning')
        handle_outliers_action = QAction('Handle Outliers', self)
        data_cleaning_menu.addAction(handle_outliers_action)
        handle_missing_action = QAction("Handle Missing Values", self)
        data_cleaning_menu.addAction(handle_missing_action)

        handle_outliers_action.triggered.connect(self.open_outliers_dialog)
        handle_missing_action.triggered.connect(self.open_missing_values_dialog)

        # Normalization menu
        normalization_menu = menubar.addMenu('Normalization')
        minmax_action = QAction('Min-Max Normalization', self)
        zscore_action = QAction('Z-score Normalization', self)
        normalization_menu.addAction(minmax_action)
        normalization_menu.addAction(zscore_action)

        minmax_action.triggered.connect(self.open_minmax_dialog)
        zscore_action.triggered.connect(self.open_zscore_dialog)

        # Discretization menu
        discretization_menu = menubar.addMenu('Discretization')
        equal_frequency_action = QAction('Equal Frequency', self)
        equal_width_action = QAction('Equal Width', self)
        discretization_menu.addAction(equal_frequency_action)
        discretization_menu.addAction(equal_width_action)

        equal_frequency_action.triggered.connect(self.open_equal_frequency_dialog)
        equal_width_action.triggered.connect(self.open_equal_width_dialog)

        # Data Reduction menu
        data_reduction_menu = menubar.addMenu('Data Reduction')
        eliminate_horizontal_action = QAction('Eliminate Horizontal Redundancies', self)
        eliminate_vertical_action = QAction('Eliminate Vertical Redundancies', self)
        data_reduction_menu.addAction(eliminate_horizontal_action)
        data_reduction_menu.addAction(eliminate_vertical_action)

        #all plot
        stats_all.triggered.connect(self.visualize_unique)


        eliminate_horizontal_action.triggered.connect(self.open_eliminate_horizontal_dialog)
        eliminate_vertical_action.triggered.connect(self.open_eliminate_vertical_dialog)


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

    def analyze_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
        dialog = AnalysisDialog(self.df, self)
        dialog.exec_()

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

    def missing_att(self,attribute, df):
        total_count = df[attribute].shape[0]

        return df[attribute].isnull().sum(), (df[attribute].isnull().sum() / total_count) * 100


    def unique_att(self, attribute, df):
        return df[attribute].nunique()
    def visualize_unique(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import a CSV file first.")
            return
        
                # Close the previous figure if it exists
        if self.figure:
            plt.close(self.figure)
        miss_uniq = {}

        for column in self.df.columns:
            miss = self.missing_att(column, self.df)
            unique = self.unique_att(column, self.df)
            miss_uniq[column] = {
            'Missing values count': miss[0],
            'Missing values percentage': miss[1],
            'Unique values count': unique
         }
        columns_numeric_to_analyze = [column for column in self.df.columns if pd.api.types.is_numeric_dtype(self.df[column])]
        dispersion_measures = {}
        for col in columns_numeric_to_analyze:
            q1 = self.df[col].quantile(q=0.25)
            q3 = self.df[col].quantile(q=0.75)
            iqr = q3 - q1
            iqr_1_5 = 1.5 * iqr

            lower_bound = q1 - iqr_1_5
            upper_bound = q3 + iqr_1_5

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            num_outliers = outliers.count()
            dispersion_measures[col] = {
            'Number of Outliers': num_outliers
            }
        dispersion_df = pd.DataFrame(dispersion_measures).T
        missuniq_df = pd.DataFrame(miss_uniq).T
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missuniq_df.index, y=missuniq_df['Missing values count'], color='salmon')
        plt.title("Nombre de Valeurs Manquantes par Attribut", fontsize=14)
        plt.xlabel("Attributs", fontsize=12)
        plt.ylabel("Nombre de Valeurs Manquantes", fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=missuniq_df.index, y=missuniq_df['Missing values percentage'], color='lightcoral')
        plt.title("Pourcentage de Valeurs Manquantes par Attribut", fontsize=14)
        plt.xlabel("Attributs", fontsize=12)
        plt.ylabel("Pourcentage de Valeurs Manquantes (%)", fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=missuniq_df.index, y=missuniq_df['Unique values count'], color='lightgreen')
        plt.title("Nombre de Valeurs Uniques par Attribut", fontsize=14)
        plt.xlabel("Attributs", fontsize=12)
        plt.ylabel("Nombre de Valeurs Uniques", fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        outliers_count = dispersion_df['Number of Outliers']

        plt.figure(figsize=(10, 6))
        plt.bar(outliers_count.index, outliers_count.values, color='skyblue')


        plt.title("Nombre d'Outliers par Attribut", fontsize=14)
        plt.xlabel("Attributs", fontsize=12)
        plt.ylabel("Nombre d'Outliers", fontsize=12)

        plt.xticks(rotation=90)

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

    def open_outliers_dialog(self):
        dialog = OutliersDialog(self.df, self.update_dataframe, self)
        dialog.exec_()

    def open_missing_values_dialog(self):
        dialog = MissingValuesDialog(self.df, self.update_dataframe, self)
        dialog.exec_()   

    def open_minmax_dialog(self):
        dialog = MinMaxDialog(self.df, self.update_dataframe_normalize, self)
        dialog.exec_()

    def open_zscore_dialog(self):
        dialog = ZScoreDialog(self.df, self.update_dataframe_normalize, self)
        dialog.exec_()

    def update_dataframe_normalize(self, column, normalized_data):
        self.df[column + '_normalized'] = normalized_data  # Add the normalized data as a new column
        model = PandasModel(self.df)  # Update the QTableView with the new DataFrame
        self.data_table.setModel(model)  # Refresh the table with the updated model

    def open_equal_frequency_dialog(self):
        dialog = EqualFrequencyDialog(self.df, self.update_dataframe_discretized, self)
        dialog.exec_()

    def open_equal_width_dialog(self):
        dialog = EqualWidthDialog(self.df, self.update_dataframe_discretized, self)
        dialog.exec_()

    def update_dataframe_discretized(self, column, discretized_data):
        self.df[column + '_discretized'] = discretized_data  # Add the discretized data as a new column
        model = PandasModel(self.df)  # Update the QTableView with the new DataFrame
        self.data_table.setModel(model)  # Refresh the table with the updated model

    def open_eliminate_horizontal_dialog(self):
        dialog = EliminateHorizontalDialog(self.df, self.update_dataframe_reduce, self)
        dialog.exec_()

    def open_eliminate_vertical_dialog(self):
        dialog = EliminateVerticalDialog(self.df, self.update_dataframe_reduce, self)
        dialog.exec_()

    def update_dataframe_reduce(self, reduced_data):
        self.df = reduced_data  # For vertical elimination, update the main DataFrame
        model = PandasModel(self.df)  # Update the QTableView with the new DataFrame
        self.data_table.setModel(model)  # Refresh the table with the updated model



from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QMessageBox



class EliminateHorizontalDialog(QDialog):
    def __init__(self, df, apply_reduction, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_reduction = apply_reduction

        self.setWindowTitle("Eliminate Horizontal Redundancies")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        layout = QVBoxLayout()

        # Apply button for vertical redundancy elimination
        apply_button = QPushButton("Apply Elimination")
        apply_button.clicked.connect(self.apply_horizontal_elimination)
        layout.addWidget(apply_button)

        self.setLayout(layout)
        
    def eliminate_horizontal_redundancies(self, df):
        return df.drop_duplicates()

    def apply_horizontal_elimination(self):
        reduced_data = self.eliminate_horizontal_redundancies(self.df)
        self.apply_reduction(reduced_data)
        self.accept()


class EliminateVerticalDialog(QDialog):
    def __init__(self, df, apply_reduction, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_reduction = apply_reduction

        self.setWindowTitle("Eliminate Vertical Redundancies")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Apply button for vertical redundancy elimination
        apply_button = QPushButton("Apply Elimination")
        apply_button.clicked.connect(self.apply_vertical_elimination)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    
    def eliminate_vertical_redundancies(self, df):
        return df.loc[:, df.nunique() > 1]


    def apply_vertical_elimination(self):
        reduced_data = self.eliminate_vertical_redundancies(self.df)
        self.apply_reduction(reduced_data)  # No column for vertical elimination
        self.accept()


class EqualFrequencyDialog(QDialog):
    def __init__(self, df, apply_discretization, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_discretization = apply_discretization

        self.setWindowTitle("Equal Frequency Discretization")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Column selection
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.df.columns)
        layout.addWidget(QLabel("Select column to discretize:"))
        layout.addWidget(self.column_combo)

        # Bins input
        self.bins_input = QLineEdit()
        self.bins_input.setPlaceholderText("Enter number of bins")
        layout.addWidget(QLabel("Number of Bins:"))
        layout.addWidget(self.bins_input)

        # Apply button
        apply_button = QPushButton("Apply Discretization")
        apply_button.clicked.connect(self.apply_equal_frequency)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def discritize_equal_frequency(self, df, column, bins):
        discretized_column = pd.qcut(df[column], q=bins, labels=False, duplicates='drop')
        return discretized_column

    def apply_equal_frequency(self):
        column = self.column_combo.currentText()
        bins = self.bins_input.text()

        # Validate input
        try:
            bins = int(bins)
            if bins <= 0:
                raise ValueError("Number of bins must be greater than 0.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
            return

        discretized_data = self.discritize_equal_frequency(self.df, column, bins)
        self.apply_discretization(column, discretized_data)
        self.accept()

class EqualWidthDialog(QDialog):
    def __init__(self, df, apply_discretization, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_discretization = apply_discretization

        self.setWindowTitle("Equal Width Discretization")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Column selection
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.df.columns)
        layout.addWidget(QLabel("Select column to discretize:"))
        layout.addWidget(self.column_combo)

        # Bins input
        self.bins_input = QLineEdit()
        self.bins_input.setPlaceholderText("Enter number of bins")
        layout.addWidget(QLabel("Number of Bins:"))
        layout.addWidget(self.bins_input)

        # Apply button
        apply_button = QPushButton("Apply Discretization")
        apply_button.clicked.connect(self.apply_equal_width)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def discritize_equal_width(self, df, column, bins):
        discretized_column = pd.cut(df[column], bins=bins, labels=False, duplicates='drop')
        return discretized_column


    def apply_equal_width(self):
        column = self.column_combo.currentText()
        bins = self.bins_input.text()

        # Validate input
        try:
            bins = int(bins)
            if bins <= 0:
                raise ValueError("Number of bins must be greater than 0.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
            return

        discretized_data = self.discritize_equal_width(self.df, column, bins)
        self.apply_discretization(column, discretized_data)
        self.accept()



class MinMaxDialog(QDialog):
    def __init__(self, df, apply_minmax, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_minmax = apply_minmax

        self.setWindowTitle("Min-Max Normalization")
        self.setGeometry(100, 100, 300, 250)  # Adjusted height for additional widgets
        
        layout = QVBoxLayout()

        # Column selection
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.df.columns)
        layout.addWidget(QLabel("Select column to normalize:"))
        layout.addWidget(self.column_combo)

        # New Min and Max inputs
        self.new_min_input = QLineEdit()
        self.new_min_input.setPlaceholderText("Enter new min value (default 0)")
        layout.addWidget(QLabel("New Min:"))
        layout.addWidget(self.new_min_input)

        self.new_max_input = QLineEdit()
        self.new_max_input.setPlaceholderText("Enter new max value (default 1)")
        layout.addWidget(QLabel("New Max:"))
        layout.addWidget(self.new_max_input)

        # Apply button
        apply_button = QPushButton("Apply Normalization")
        apply_button.clicked.connect(self.apply_minmax_normalization)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def minmax_normalization(self, df, column, new_min=0, new_max=1):
        old_min = df[column].min()
        old_max = df[column].max()

        normalized_column = ((df[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return normalized_column

    def apply_minmax_normalization(self):
        column = self.column_combo.currentText()
        new_min = self.new_min_input.text()
        new_max = self.new_max_input.text()

        # Validate input
        try:
            new_min = float(new_min) if new_min else 0  # Default to 0 if empty
            new_max = float(new_max) if new_max else 1  # Default to 1 if empty
            if new_min >= new_max:
                raise ValueError("New Min must be less than New Max.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
            return
        
        normalized_data = self.minmax_normalization(self.df, column, new_min, new_max)
        self.apply_minmax(column, normalized_data)
        self.accept()

from scipy.stats import zscore

class ZScoreDialog(QDialog):
    def __init__(self, df, apply_zscore, parent=None):
        super().__init__(parent)
        self.df = df
        self.apply_zscore = apply_zscore

        self.setWindowTitle("Z-Score Normalization")
        self.setGeometry(100, 100, 300, 200)
        
        layout = QVBoxLayout()
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.df.columns)
        layout.addWidget(QLabel("Select column to normalize:"))
        layout.addWidget(self.column_combo)

        apply_button = QPushButton("Apply Normalization")
        apply_button.clicked.connect(self.apply_zscore_normalization)
        layout.addWidget(apply_button)

        self.setLayout(layout)



    def zscore_normalization(self, df, column):
        normalized_column = zscore(df[column])
        return normalized_column


    def apply_zscore_normalization(self):
        column = self.column_combo.currentText()
        normalized_data = self.zscore_normalization(self.df, column)
        self.apply_zscore(column, normalized_data)
        self.accept()


class OutliersDialog(QDialog):
    def __init__(self, df, update_callback, parent=None):
        super().__init__(parent)
        self.df = df
        self.update_callback = update_callback

        self.setWindowTitle("Handle Outliers")
        self.setGeometry(300, 300, 400, 300)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select the method for handling outliers:"))
        self.method_selector = QComboBox()
        self.method_selector.addItems(["IQR Method", "Z-Score Method",  "Mean Replacement", "Median Replacement", "KNN Replacement"])
        layout.addWidget(self.method_selector)

        self.column_selector = QComboBox()
        self.column_selector.addItems(df.columns)
        layout.addWidget(QLabel("Select column for outlier removal:"))
        layout.addWidget(self.column_selector)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.handle_outliers)
        layout.addWidget(self.process_button)

    def handle_outliers(self):
        selected_method = self.method_selector.currentText()
        column = self.column_selector.currentText()

        if selected_method == "IQR Method":
            new_df = self.remove_outliers_iqr(self.df, column)
            QMessageBox.information(self, "Info", "IQR Outliers method applied.")
        
        elif selected_method == "Z-Score Method":
            threshold, ok = QInputDialog.getDouble(self, "Input Threshold", "Enter the Z-Score threshold (default is 3):", value=3.0)
            if ok:
                new_df = self.remove_outliers_zscore(self.df, column, threshold)
                QMessageBox.information(self, "Info", "Z-Score Outliers method applied.")
        #elif selected_method == "Mean Replacement":
        elif selected_method == 'Mean Replacement':
            new_df = self.replace_outliers_mean(self.df, column)
            QMessageBox.information(self, "Info", "Mean Replacement method applied.")
        elif selected_method == "Median Replacement":
            new_df = self.replace_outliers_median(self.df, column)
            QMessageBox.information(self, "Info", "Median Replacement method applied.")
        elif selected_method == "KNN Replacement":
            k, ok = QInputDialog.getDouble(self, "Input nb neighbour", "Enter the nomber of neighbour  (default is 5):", value=5)
            if ok:
                new_df = self.replace_outliers_knn(self.df, column, int(k))
                QMessageBox.information(self, "Info", "KNN Replacement method applied.")



        
        # Call the update callback to refresh the main DataFrame in the UI
        self.update_callback(new_df)
        self.close()

    def remove_outliers_iqr(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def remove_outliers_zscore(self, df, column, threshold=3):
        mean = df[column].mean()
        std_dev = df[column].std()
        z_scores = (df[column] - mean) / std_dev
        return df[abs(z_scores) <= threshold]
    
    def replace_outliers_mean(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mean_value = df[column].mean() 
        df[column] = df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
        return df
    
    
    def replace_outliers_median(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        median_value = df[column].median()

        df[column] = df[column].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)
        return df

    def replace_outliers_knn(self, df, column, nb_neighbors=5):
        Q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        normal_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        if outliers.empty:
            return df
    
        featurs = [col for col in df.columns if col != column]
        X_train = normal_data[featurs]
        y_train = normal_data[column]
    
        X_outliers = outliers[featurs]

        knn = KNeighborsRegressor(n_neighbors=nb_neighbors)
        knn.fit(X_train, y_train)
        predicated_values = knn.predict(X_outliers)
        df.loc[X_outliers.index, column] = predicated_values
        return df

class MissingValuesDialog(QDialog):
    def __init__(self, df, update_callback, parent=None):
        super().__init__(parent)
        self.df = df
        self.update_callback = update_callback

        self.setWindowTitle("Handle Missing Values")
        self.setGeometry(300, 300, 400, 300)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select the method for handling missing values:"))
        self.method_selector = QComboBox()
        self.method_selector.addItems([
            "Drop Rows with Any NA", "Drop Rows with All NA", "Drop Rows with Threshold",
            "Drop Columns with Any NA", "Drop Columns with All NA",
            "Fill with Value", "Fill with Mean", "Fill with Median", 
            "Fill Forward", "Fill Backward"
        ])
        layout.addWidget(self.method_selector)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.handle_missing_values)
        layout.addWidget(self.process_button)

    def handle_missing_values(self):
        selected_method = self.method_selector.currentText()

        if selected_method == "Drop Rows with Any NA":
            new_df = self.drop_rows_na(self.df)
            QMessageBox.information(self, "Info", "Rows with any NA dropped.")
        
        elif selected_method == "Drop Rows with All NA":
            new_df = self.drop_rows_all_na(self.df)
            QMessageBox.information(self, "Info", "Rows with all NA dropped.")
        
        elif selected_method == "Drop Rows with Threshold":
            threshold, ok = QInputDialog.getInt(self, "Input Threshold", "Enter the threshold value:")
            if ok:
                new_df = self.drop_rows_thresh_na(self.df, threshold)
                QMessageBox.information(self, "Info", f"Rows with less than {threshold} non-NA values dropped.")

        elif selected_method == "Drop Columns with Any NA":
            new_df = self.drop_columns_na(self.df)
            QMessageBox.information(self, "Info", "Columns with any NA dropped.")
        
        elif selected_method == "Drop Columns with All NA":
            new_df = self.drop_columns_all_na(self.df)
            QMessageBox.information(self, "Info", "Columns with all NA dropped.")

        elif selected_method == "Fill with Value":
            value, ok = QInputDialog.getText(self, "Input Value", "Enter the value to fill missing data with:")
            if ok:
                new_df = self.fill_na(self.df, value)
                QMessageBox.information(self, "Info", "Missing values filled with the given value.")

        elif selected_method == "Fill Forward":
            limit, ok = QInputDialog.getInt(self, "Input Limit", "Enter the limit for forward fill (optional, press OK for no limit):", value=0, min=0)
            if ok:
                new_df = self.fill_forward_na(self.df, limit if limit > 0 else None)
                QMessageBox.information(self, "Info", "Missing values forward filled.")

        elif selected_method == "Fill Backward":
            limit, ok = QInputDialog.getInt(self, "Input Limit", "Enter the limit for backward fill (optional, press OK for no limit):", value=0, min=0)
            if ok:
                new_df = self.fill_backward_na(self.df, limit if limit > 0 else None)
                QMessageBox.information(self, "Info", "Missing values backward filled.")

        elif selected_method == "Fill with Mean":
            new_df = self.fill_mean_na(self.df)
            QMessageBox.information(self, "Info", "Missing values filled with mean.")

        elif selected_method == "Fill with Median":
            new_df = self.fill_median_na(self.df)
            QMessageBox.information(self, "Info", "Missing values filled with median.")

        # Call the update callback to refresh the main DataFrame in the UI
        self.update_callback(new_df)
        self.close()

    # Implementations for missing value handling methods
    def drop_rows_na(self, df):
        return df.dropna()

    def drop_rows_all_na(self, df):
        return df.dropna(how='all')

    def drop_rows_thresh_na(self, df, th):
        return df.dropna(thresh=th)

    def drop_columns_na(self, df):
        return df.dropna(axis="columns")

    def drop_columns_all_na(self, df):
        return df.dropna(axis="columns", how='all')

    def fill_na(self, df, value):
        return df.fillna(value)

    def fill_forward_na(self, df, limit=None):
        return df.fillna(method="ffill", limit=limit)

    def fill_backward_na(self, df, limit=None):
        return df.fillna(method="bfill", limit=limit)

    def fill_mean_na(self, df):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
        return df

    def fill_median_na(self, df):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].median(), inplace=True)
        return df


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

    
    def scatter_plot(self, df,column_x, column_y):
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


    def create_scatter_plot(self):
        column_x = self.column_x_selector.currentText()
        column_y = self.column_y_selector.currentText()
        if (self.df[column_x].dtype and self.df[column_y].dtype) in [np.float64, np.int64]:
            self.scatter_plot(self.df, column_x, column_y)  # Assuming scatter_plot function is defined globally
            self.close()
        else:
            QMessageBox.warning(self, "Warning", "Selected column is not numeric.")


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
