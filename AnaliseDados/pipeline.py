import joblib
import numpy as np
import matplotlib.pyplot as plt


pipeline = joblib.load("results/pipeline_objects.joblib")


print("Chaves disponíveis:", pipeline.keys())

scaler = pipeline['scaler']
pca = pipeline['pca']
features = pipeline['features']

print("\nNúmero de componentes PCA:", len(pca.explained_variance_ratio_))
print("Variância explicada acumulada (300 componentes):", np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Número de componentes")
plt.ylabel("Variância explicada acumulada")
plt.title("Curva de variância PCA")
plt.grid(True)
plt.tight_layout()
plt.show()


comp0 = pca.components_[0]
top_idx = np.argsort(np.abs(comp0))[::-1][:10]
print("\nTop 10 features que mais influenciam o 1º componente PCA:")
for i in top_idx:
    print(f"{features[i]} -> peso {comp0[i]:.4f}")
