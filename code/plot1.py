import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../datasets/diabetes.csv")

f = plt.figure(figsize=(8,4))
ax = f.add_subplot(1,1,1)
data["Age"].hist(ax=ax, bins=5, edgecolor='black', linewidth=2)
ax.set_title("Age range of patients")
ax.set_ylim([0, 510])
ax.set_xlabel("Age")
ax.set_ylabel("Count")
f.tight_layout()

plt.show()