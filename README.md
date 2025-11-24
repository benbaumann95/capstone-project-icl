# Capstone Project – Black-Box Optimization Challenge

This repository contains the code and analysis for the **Black-Box Optimization (BBO) Challenge**, part of the Imperial College Professional Certificate in Machine Learning & Artificial Intelligence.

The goal is to optimize 8 unknown "black-box" functions over several weeks by iteratively proposing query points and learning from the results.

## Project Structure

```text
├── data/                     # Contains samples.csv for each function (initial + new data)
├── notebooks/                # Jupyter notebooks for weekly analysis and query generation
├── src/                      # Source code for reusable logic
│   ├── utils.py              # Helper functions (data loading, submission logging)
│   └── initialize_samples.py # Script to reset/init data from .npy files
├── submissions/              # Log of submitted queries
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository**:

    ```bash
    git clone <repo-url>
    cd capstone-project-icl
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize Data** (if starting fresh):
    ```bash
    python src/initialize_samples.py
    ```

## Usage

### Weekly Workflow

1.  **Run the Weekly Notebook**:

    - Open `notebooks/01_Module_12.ipynb` (for Week 1).
    - Run all cells to generate queries for Functions 1-8.
    - The notebook uses an adaptive Bayesian Optimization strategy:
      - **Function 1**: High Exploration (Uncertainty Sampling) due to sparse data.
      - **Functions 2-8**: Balanced Exploration/Exploitation (Standard UCB).

2.  **Submit Queries**:

    - Copy the generated queries from the notebook output or `submissions/submission_log.csv`.
    - Submit them to the challenge portal.

3.  **Update Data**:
    - After receiving new results, update the `samples.csv` files in `data/function_X/`.

## Modules

### Module 12: Initial Submission

- **Goal**: Generate the first query point for all 8 functions.
- **Strategy**: Gaussian Process Regression with Matern Kernel.
  - **Acquisition Function**: Upper Confidence Bound (UCB).
  - **Key Insight**: Function 1 requires aggressive exploration (`kappa=10.0`) as initial samples are all zero. Other functions use standard settings (`kappa=1.96`).

1.  **Update Data**: Add the new data point received from the portal to the corresponding `data/function_X/samples.csv`.
2.  **Run Analysis**: Create or update a notebook in `notebooks/` (e.g., `01_Module_12.ipynb`) to analyze the new data.
3.  **Generate Query**: Use the notebook to propose the next query point.
4.  **Submit**: Enter the query into the Capstone Portal.
5.  **Reflect**: Document the strategy and reasoning in the notebook.

## Methodology

The core approach uses **Bayesian Optimization** with **Gaussian Processes (GPs)**.

- **Model**: Gaussian Process Regressor (scikit-learn).
- **Acquisition Function**: Upper Confidence Bound (UCB) to balance exploration (uncertainty) and exploitation (high predicted values).
- **Strategy**:
  - _Initial Phase_: High exploration (Uncertainty Sampling) to map the space.
  - _Later Phase_: Exploitation to refine the maximum.

## Author

Benjamin Baumann
