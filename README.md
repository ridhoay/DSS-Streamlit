# Decision Support System Streamlit App

This repository contains a **Streamlit web application** designed to help with decision-making using a Decision Support System (DSS).

---

## Files Overview

* `main.py`: The main Streamlit application file. Run this to launch the web app.
* `dss_engine.py`: Contains the core logic for the Decision Support System.
* `data_raw.csv`: **Sample input data** that users can upload to the DSS for analysis.
* `pw_main.csv`: A sample **pairwise comparison matrix**, likely used for Analytic Hierarchy Process (AHP) weighting within the DSS.
* `requirements.txt`: Lists the necessary Python packages to run the application.
* `converter.ipynb`: (Optional) A Jupyter Notebook that might have been used for data conversion or preliminary analysis.
* `__pycache__/dss_engine.cpython-311.pyc`: Python bytecode cache.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
    This will open the application in your web browser.

---

## Usage

* Upload your own data or use the provided `data_raw.csv` as an example.
* The application may utilize the `pw_main.csv` for weighting criteria based on pairwise comparisons.
* Follow the on-screen instructions within the Streamlit app to interact with the Decision Support System.
