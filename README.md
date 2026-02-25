# Customer Segmentation Dashboard

A production-ready customer segmentation system built using **RFM analysis (Recency, Frequency, Monetary)** and **KMeans clustering**.  

This project extends a customer segmentation model built using KMeans clustering into a complete end-to-end application with:

- FastAPI REST API
- Streamlit interactive dashboard
- Docker-based deployment

---

## 📌 Project Overview

This project segments customers based on:

- **Recency** – Days since last purchase  
- **Frequency** – Number of purchases  
- **Monetary Value** – Total amount spent  

The features are scaled using `StandardScaler` and clustered using `KMeans`.

The trained model is wrapped inside an API and connected to an interactive dashboard for easy testing and visualization.

---

## Dashboard Features

- Upload external CSV file
- Automatically detects:
  - RFM table
  - Raw transaction dataset
- Computes RFM
- Predicts customer segments
- Displays:
  - Recency distribution by cluster
  - Frequency distribution by cluster
  - Monetary distribution by cluster

## Use Cases

- Customer segmentation analysis
- Marketing targeting
- Customer retention strategies
  
