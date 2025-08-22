
# Lung Cancer Survival Prediction using Machine Learning

## Project Overview
This project aims to predict 1-year survival outcomes for lung cancer patients using a synthetic dataset from a Bangladesh perspective. The dataset contains 5000 records with 17 features, including patient demographics, clinical details, and environmental factors. The target variable is `Survival_1_Year` (Yes/No), indicating whether a patient survives one year post-diagnosis. The project involves data exploration, preprocessing, feature engineering, handling imbalanced data with SMOTE, training multiple machine learning models, and implementing a hybrid model to improve prediction accuracy.

## Dataset
- **Source**: `Large_Synthetic_Lung_Cancer_Dataset__Bangladesh_Perspective_.csv`
- **Size**: 5000 rows, 17 columns
- **Features**: `Patient_ID`, `Age`, `Gender`, `Smoking_Status`, `Residence`, `Air_Pollution_Exposure`, `Biomass_Fuel_Use`, `Factory_Exposure`, `Family_History`, `Diet_Habit`, `Symptoms`, `Tumor_Size_mm`, `Histology_Type`, `Stage`, `Treatment`, `Hospital_Type`, `Survival_1_Year`
- **Target**: `Survival_1_Year` (Yes/No)
- **Class Distribution**: Imbalanced (Yes: 3491, No: 1509)

## Project Structure
1. **Data Loading and Exploration**:
   - Loaded dataset using `pandas`.
   - Checked for missing values (none found).
   - Analyzed target variable distribution using `sns.countplot`.
   - Performed feature importance analysis with `RandomForestClassifier`.

2. **Statistical Analysis**:
   - Conducted Chi-squared test to check association between `Hospital_Type` and `Survival_1_Year`.

3. **Data Preprocessing**:
   - Dropped `Patient_ID` as it’s non-informative.
   - Encoded categorical features using `LabelEncoder`.
   - Applied SMOTE to balance the dataset (`imbalanced-learn`).

4. **Feature Engineering**:
   - Created `Age_Tumor_Interaction` (`Age` * `Tumor_Size_mm`).
   - Combined `Smoking_Status` and `Air_Pollution_Exposure` into `Smoking_Pollution_Combined`.

5. **Model Training**:
   - Split data into 80-20 train-test sets.
   - Trained models: `RandomForestClassifier`, `DecisionTreeClassifier`, `SVC`, `LogisticRegression`, `KNeighborsClassifier`, `XGBoost`.
   - Evaluated using accuracy, confusion matrix, and classification report.

6. **Hybrid Model**:
   - Used `VotingClassifier` with hard and soft voting (`RandomForest`, `DecisionTree`, `LogisticRegression`, `KNeighborsClassifier`).

7. **Model Comparison**:
   - Plotted accuracy comparison using a bar chart.
   - Best accuracy: `SVC` and `LogisticRegression` (0.694).

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Required libraries:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `imbalanced-learn`
   - `matplotlib`
   - `seaborn`
   - `xgboost`
   - `scipy`

## Usage
1. Place the dataset (`Large_Synthetic_Lung_Cancer_Dataset__Bangladesh_Perspective_.csv`) in the project directory.
2. Run the Jupyter notebook (`lung_cancer__(2).ipynb`) to execute the code.
3. Outputs include:
   - Data exploration visualizations (e.g., count plots, heatmaps).
   - Feature importance bar chart.
   - Confusion matrices and classification reports for each model.
   - Accuracy comparison bar chart.

## Results
- **Best Performing Models**: `SVC` and `LogisticRegression` (Accuracy: 0.694).
- **Challenges**: Poor performance on the minority class (`No`) due to imbalanced data, even after SMOTE.
- **Recommendations**: Further improve accuracy with hyperparameter tuning, advanced feature engineering, or more complex models like Gradient Boosting.

## Future Improvements
- Perform hyperparameter tuning using `GridSearchCV`.
- Explore additional feature interactions or domain-specific features.
- Test advanced ensemble methods like stacking or boosting.

## License
This project is licensed under the MIT License.
