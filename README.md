# Healthcare Outcome Prediction - End-to-End MLOps Pipeline
This project implements a robust, automated Machine Learning lifecycle designed to process patient data and predict healthcare outcomes. It focuses heavily on MLOps best practices, ensuring reproducibility, tracking, and seamless model artifact generation.

## 🛠 Tech Stack

#### Language: Python 3.x

#### Machine Learning: Scikit-Learn / Pandas

#### Tracking/Ops: MLflow & DVC (Data Version Control)

#### Infrastructure: Docker (Deployment ready)

## 🏗 ML Pipeline Workflows
The project is structured into modular components to ensure clean version control and isolated execution:

*Data Ingestion*: Fetching and loading the synthetic healthcare outcome dataset.

*Data Validation*: Ensuring data quality, checking schema consistency, and validating data types (e.g., ensuring `Billing Amount` is properly formatted as a float and dates are parsed correctly).

*Data Transformation*: Feature engineering, handling missing values, encoding categorical variables (like `Medical Condition` and `Medication`), and preparing the dataset for model training.

*Model Trainer*: Training the multi-category classification model and generating serialized model artifacts.

*Model Evaluation*: Logging performance metrics and registering model versions with MLflow.

### 🔄 Development Workflow (Step-by-Step)
To add a new feature or update a component, follow this specific order:

1. Update `config.yaml`: Define new paths or stage-specific configurations.
2. Update `schema.yaml`: Define column names and data types (Critical for healthcare data).
3. Update `params.yaml`: Adjust model hyperparameters.
4. Update the `Entity`: Define the return types for your configuration functions.
5. Update `Configuration Manager`: Located in `src/config/configuration.py`.
6. Update the `Components`: Write the actual logic (e.g., a new transformation step).
7. Update the `Pipeline`: Link the component to a pipeline stage.
8. Update `main.py`: Trigger the stage execution.

---

## 📊 Dataset Reference
The model is trained using a synthetic **Healthcare Dataset** designed to mirror real-world patient records for multi-category classification modeling. This data enables the practice of data manipulation and analysis in a healthcare context without exposing sensitive patient information. 

* **Dataset Link:** [Healthcare Dataset (Kaggle)](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
* **Attributes:** `Name`, `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Date of Admission`, `Doctor`, `Hospital`, `Insurance Provider`, `Billing Amount`, `Room Number`, `Admission Type`, `Discharge Date`, `Medication`, and `Test Results`.

---

## 🚀 How to Run
1. Environment Setup
```bash
conda create -n healthcare-mlops python=3.8 -y
conda activate healthcare-mlops

pip install -r requirements.txt
