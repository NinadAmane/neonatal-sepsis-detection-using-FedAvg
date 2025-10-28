# Federated Sepsis Machine Learning Project

This project implements a federated learning approach for predicting sepsis using machine learning techniques. The dataset is split into multiple parts to simulate different hospitals, allowing for decentralized training and evaluation of models.

## Project Structure

```
federated-sepsis/
├── data/
│   ├── raw/                  # Original files from Kaggle
│   │   └── prediction-of-sepsis.csv
│   ├── processed/            # Cleaned/processed CSVs
│   │   └── clean_sepsis.csv
│   └── federated/            # Split datasets for each simulated hospital/client
│       ├── hospital_1.csv
│       ├── hospital_2.csv
│       └── hospital_3.csv
│
├── scripts/                  # Utility scripts to setup the project
│   ├── download_dataset.py    # Downloads Kaggle dataset & unzips it
│   ├── preprocess_data.py     # Cleans raw CSV → processed CSV
│   └── split_federated_data.py # Splits processed CSV into multiple “hospital” CSVs
│
├── src/                      # Core ML + FL code
│   ├── train_local_model.py   # Train a local model on a single hospital
│   ├── federated_averaging.py  # Aggregate model weights for FL simulation
│   ├── evaluation.py          # Evaluate global & local models
│   └── utils.py              # Helper functions (metrics, plotting, etc.)
│
├── main.py                   # Orchestrates the full workflow
├── requirements.txt          # Dependencies
└── README.md                 # Project overview & instructions
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd federated-sepsis
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Run the following script to download the original dataset from Kaggle:
   ```
   python scripts/download_dataset.py
   ```

4. **Preprocess the data**:
   Clean the raw dataset and generate the processed CSV:
   ```
   python scripts/preprocess_data.py
   ```

5. **Split the data for federated learning**:
   Create separate datasets for each simulated hospital:
   ```
   python scripts/split_federated_data.py
   ```

## Usage

To train a local model on a specific hospital's data, run:
```
python src/train_local_model.py --hospital <hospital_number>
```

To perform federated averaging of models from multiple hospitals, use:
```
python src/federated_averaging.py
```

To evaluate the models, execute:
```
python src/evaluation.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.