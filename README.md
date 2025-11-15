# 🌩️ Smart Grid Simulation & ML-Based Anomaly Detection  
A Python-based system for synthetic electrical load simulation, anomaly detection, and event-driven fault handling inspired by operating system scheduling and interrupt mechanisms.

---

## 🚀 Overview  
This project simulates a realistic electrical time-series dataset, builds a machine learning pipeline for anomaly detection, and implements an event-driven scheduling engine for efficient fault-handling.  

Core areas covered:
- Synthetic time-series simulation  
- Automated feature engineering  
- ML-based anomaly classification  
- Event-driven OS-like fault processing  
- Performance optimization  

---

## ⚡ Key Features

### **1️⃣ Synthetic Time-Series Generation**  
A fully configurable generator that produces minute-level electrical data such as:
- Voltage  
- Current  
- Real power  
- Power factor  
- Frequency  
- Harmonic distortion  

With anomalies injected including:
- Power spike  
- Sudden drop  
- Drift  
- Noise bursts  

Feature engineering includes:
- Rolling means  
- Rolling standard deviations  
- Power percentage changes  
- Current differential features  

The dataset is created using both sinusoidal load modeling and stochastic noise patterns to resemble real industrial load behavior.

---

### **2️⃣ Machine Learning-Based Anomaly Detection**  
Two ML pipelines are implemented:

#### **Unsupervised Model (IsolationForest)**
- Learns normal patterns  
- Detects anomalies based on contamination level  
- Good for unlabeled time-series  

#### **Supervised Model (RandomForestClassifier)**
- Trained on labeled anomalies  
- Provides strong precision/recall  
- Feature importance for interpretability  

Outputs include:
- Trained model files (`joblib`)  
- Metrics JSON (precision, recall, F1)  
- Visualizations (via notebooks)

---

### **3️⃣ Event-Driven Fault Handling (OS Scheduling Inspired)**  
A lightweight event engine simulates the behavior of OS-level scheduling:

- Priority queue for event dispatch  
- Interrupt-like handling when sudden power deviations occur  
- No constant polling or busy-wait loops  
- Faster response time and lower CPU overhead  

This design reduces total processing time by avoiding unnecessary computations and handling only critical events.

---

### **4️⃣ Full Notebook Suite for Analysis**  
Three Jupyter notebooks provide:
- Dataset generation  
- ML model training  
- Full exploratory analysis  

Including:
- Time-series plots  
- Anomaly overlays  
- Correlation heatmaps  
- Distribution visualizations  
- Rolling feature analysis  
- Zoomed-in windows around detected anomalies  

---

## 🛠️ Installation  

Install required dependencies:

