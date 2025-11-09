# CTU-13 Cybersecurity Project

A cybersecurity project based on the **CTU-13 network traffic dataset**, focusing on data exploration, feature analysis, and basic machine learning experiments for malicious traffic detection.  
This repository was developed collaboratively as part of an academic assignment.

---

## 📁 Project Structure

cybersecurity-ctu13/
│
├── src/
│   └── ctu13/
│       ├── main.py
│       ├── futureselection.py
│       ├── futureselection2.py
│       ├── fromstart.py
│       ├── git.py
│       ├── git2.py
│       ├── last.py
│       ├── last2.py
│       └── merged.py
│
├── reports/
│   └── Project_Report.pdf
│
├── data/                
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 🔍 Overview

The project uses the **CTU-13 dataset** — a collection of real network traffic captured at the Czech Technical University in Prague.  
It contains **normal and malicious (botnet) traffic** and is widely used for **network intrusion detection** research.

The main goals of this project are:
- Perform exploratory data analysis (EDA) on CTU-13 flows.
- Understand key traffic features such as `StartTime`, `Dur`, `Proto`, `SrcAddr`, `DstAddr`, `TotPkts`, `TotBytes`, `Label`, etc.  
- Implement preprocessing and feature selection scripts.
- Experiment with basic machine learning models for classification.
- Summarize results and insights in the project report.

All details, feature descriptions, and visualizations are documented in the accompanying report (`reports/Project_Report.pdf`). :contentReference[oaicite:0]{index=0}

---

## 🧩 Scripts Description

| Script | Description |
|--------|--------------|
| `main.py` | Central pipeline controller for data loading and analysis. |
| `fromstart.py` | Data preprocessing and initial setup. |
| `futureselection.py`, `futureselection2.py` | Feature selection experiments. |
| `merged.py` | Merges processed datasets or analysis outputs. |
| `last.py`, `last2.py` | Final testing or evaluation scripts. |
| `git.py`, `git2.py` | Auxiliary or version helper scripts. |

---

## ⚙️ Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/BeyzaAkgun/cybersecurity-ctu13.git
   cd cybersecurity-ctu13


Create a virtual environment and install dependencies

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


Prepare the dataset

Download the CTU-13 dataset (see references in the PDF report). 

Place it in the data/ directory 

Run the main script

python src/ctu13/main.py --data-path ./data/sample.binetflow

🧑‍💻 Authors

Beyza Akgün

Ahmet Yiğit Özkoca

Yusuf Eskiocak

📚 References

Project Report: reports/Project_Report.pdf
Includes dataset description, feature list, graphs, and related works. 
Dataset: CTU-13 — Stratosphere Lab, Czech Technical University in Prague.
(https://www.stratosphereips.org/datasets-ctu13)
