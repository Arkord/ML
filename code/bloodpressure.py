import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../datasets/diabetes.csv")
data = data[data["BloodPressure"] > 0]

plt.rcParams['font.size'] = 15 
f = plt.figure(figsize=(8,4))
ax = f.add_subplot(1,1,1)
ax.scatter(data["Age"], data["BloodPressure"], alpha=0.25)
ax.set_title("Blood pressure vs. age range of patients")
ax.set_ylabel("Blood pressure")
ax.set_xlabel("Age")
f.tight_layout() 

plt.show()

