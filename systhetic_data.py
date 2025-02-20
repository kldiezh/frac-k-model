import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


num_samples = 500

eo_range = (9.8425e-5, 9.8425e-5)
Sn_range = (1000, 4000)
kccf = 50
Z_range = (0.2, 0.8) # krcf/kccf (Sensitivity to variation in stresses)
kni_range = (1.0e6, 1.0e6)

eo_values = np.random.uniform(*eo_range, num_samples)
Sn_values = np.random.uniform(*Sn_range, num_samples)
Z_values = np.random.uniform(*Z_range, num_samples)
kni_values = np.random.uniform(*kni_range, num_samples)

krcf_values = Z_values * kccf
Vm_values = eo_values * (1 - (krcf_values / kccf) ** (1/4))
Vj_values = Sn_values / (kni_values + (Sn_values / Vm_values))
e_values = eo_values - Vj_values
kf_values = kccf * (e_values / eo_values) ** 4

data = pd.DataFrame({
    'eo': eo_values*304800,
    'Sn': Sn_values,
    'Z': Z_values,
    'kni': kni_values,
    'krcf': krcf_values,
    'Vm': Vm_values,
    'Vj': Vj_values,
    'e': e_values*304800,
    'kf': kf_values
})

# Save the generated data to a CSV file
data.to_csv("synthetic_data_csv.csv", index=False)

# Save the generated data to a xlsx file
data.to_excel("synthetic_data_excel.xlsx", index=False)

print("Synthetic data generated and saved successfully!")

