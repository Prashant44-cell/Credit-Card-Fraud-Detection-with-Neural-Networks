import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
np.random.seed(42)
n_customers = 500

data = pd.DataFrame({
    'Duration': np.random.randint(1, 60, n_customers),
    'Churned': np.random.choice([0, 1], size=n_customers, p=[0.6, 0.4]),
    'Monthly_Charges': np.random.uniform(30, 100, n_customers),
    'Contract_Type': np.random.choice(['Monthly', 'Annual'], size=n_customers, p=[0.7, 0.3])
})


data.loc[data['Churned'] == 0, 'Duration'] = data.loc[data['Churned'] == 0, 'Duration'] + 10
data = pd.get_dummies(data, columns=['Contract_Type'], drop_first=True)
data.rename(columns={'Contract_Type_Monthly': 'Is_Monthly_Contract'}, inplace=True)
data_for_model = data[['Duration', 'Churned', 'Monthly_Charges', 'Is_Monthly_Contract']]

cph = CoxPHFitter()
cph.fit(data_for_model, duration_col='Duration', event_col='Churned', show_progress=False)

# Define two customer profiles for comparison
new_customers = pd.DataFrame([
    {'Monthly_Charges': 95, 'Is_Monthly_Contract': 1}, # High Risk (Monthly, High Charge)
    {'Monthly_Charges': 40, 'Is_Monthly_Contract': 0}  # Low Risk (Annual, Low Charge)
])

cph.plot_partial_effects_on_outcome(covariates='Is_Monthly_Contract', values=[0, 1], figsize=(10, 6))
plt.title('Predicted Survival Probability by Contract Type')
plt.xlabel('Time (Months)')
plt.ylabel('Survival Probability (1 - Churn Probability)')
plt.show() 

print("\n--- Model Summary ---")
# FIX: Ensures the function call is closed properly
cph.print_summary()
 




