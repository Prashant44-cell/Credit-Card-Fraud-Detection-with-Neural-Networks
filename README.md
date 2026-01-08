# Credit-Card-Fraud-Detection-with-Neural-Networks

# 📊 Customer Churn Survival Analysis

## 📌 Project Overview
This project applies **survival analysis** to customer churn prediction using the **Cox Proportional Hazards Model** from the `lifelines` library. Unlike traditional classification approaches, survival analysis models **time-to-event data** — here, the time until a customer churns.


---
## 🔑 Key Objectives
- Simulate synthetic customer data with:
  - **Duration** (time until churn or censoring)
  - **Churned** (event indicator: 1 = churned, 0 = active)
  - **Monthly Charges** (continuous predictor)
  - **Contract Type** (categorical predictor: Monthly vs Annual)
- Train a **Cox Proportional Hazards model** to estimate hazard ratios.
- Compare survival probabilities between different customer profiles.
- Visualize survival curves to interpret churn risk.

---

## ⚙️ Workflow
1. **Data Generation**  
   Synthetic dataset of 500 customers is created with randomized attributes. Non-churned customers are given longer durations to simulate censoring.

2. **Feature Engineering**  
   - Contract type converted into a binary variable (`Is_Monthly_Contract`).  
   - Final dataset includes duration, churn status, monthly charges, and contract type.

3. **Model Training**  
   - CoxPHFitter is trained using duration and churn status.  
   - Covariates: `Monthly_Charges`, `Is_Monthly_Contract`.

4. **Customer Profile Comparison**  
   - **High Risk**: Monthly contract, high charges.  
   - **Low Risk**: Annual contract, low charges.  
   - Survival curves plotted for comparison.

5. **Visualization & Interpretation**  
   - Survival probability curves show how contract type impacts retention.  
   - Model summary provides hazard ratios and statistical significance.

---

## 📈 Insights
- Customers with **monthly contracts** and **higher charges** churn earlier.  
- Customers with **annual contracts** and **lower charges** survive longer (lower churn risk).  
- The Cox model provides interpretable hazard ratios, useful for business decision-making.

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas / NumPy** for data handling  
- **lifelines** for survival analysis  
- **Matplotlib** for visualization  

---


