from bayes_opt import BayesianOptimization
from model import run  # Simülasyon fonksiyonunu içe aktarıyoruz
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time  # Zaman ölçümü için
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Başlangıç zamanını kaydedelim
start_time = time.time()

# ----------------------------
# Amaç Fonksiyonu (Multi-Run Versiyonu)
def objective_function(reorder_point, order_size):
    costs = []
    # Aynı parametre kombinasyonu ile 5 kez simülasyonu çalıştırıp ortalama total cost hesaplıyoruz
    for _ in range(5):
        result = run(reorder_point, order_size)
        costs.append(result['total_cost'])
    avg_cost = np.mean(costs)
    return -avg_cost  # Bayesyen optimizasyon maximize ettiğinden, minimize etmek için negatifini alıyoruz

# ----------------------------
# Karar Değişkenleri
pbounds = {
    'reorder_point': (0, 100),
    'order_size': (5, 50)
}

# ----------------------------
# Bayesian Optimization Nesnesini Oluşturma
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)

# ----------------------------
# Optimizasyonu Gerçekleştirme: 
# 5 başlangıç noktası ve 100 iterasyon (n_iter) kullanılıyor
optimizer.maximize(init_points=5, n_iter=100)

# ----------------------------
# Kümülatif En İyi (Minimum) Cost Grafiğini Çizdirme
# Bayesyen optimizasyonda objective fonksiyonun çıktısı -avg_cost olduğundan, gerçek cost = -target
results = optimizer.res  # Her iterasyonun sonuçlarını içeren liste (her eleman bir sözlük)
costs = [-res['target'] for res in results]  # Her iterasyondaki gerçek total cost değerlerini elde ediyoruz

# Kümülatif minimum: Her iterasyonda şimdiye kadar elde edilen en iyi (minimum) cost değeri
cum_min = []
current_min = float('inf')
for cost in costs:
    if cost < current_min:
        current_min = cost
    cum_min.append(current_min)

iterations = np.arange(1, len(cum_min) + 1)

plt.figure(figsize=(10, 6))
plt.plot(iterations, cum_min, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Minimum Total Cost')
plt.title('Cumulative Minimum Total Cost over Bayesian Optimization Iterations')
plt.grid(True)
plt.show()

# ----------------------------
# Iterasyon Sonuçlarının Tablo Şeklinde Yazdırılması
# Yeni en iyi değer bulunduğunda mor renkle vurgulansın.
print("\nIteration Results:")
print("{:<10s}{:<20s}{:<20s}".format("Iteration", "Cost", "Cumulative Min"))
purple = "\033[95m"
reset = "\033[0m"
for i, cost in enumerate(costs, start=1):
    if cost == cum_min[i-1]:
        # Yeni en iyi değer bulunduğunda mor renk
        print("{:<10d}{}{: <20.2f}{: <20.2f}{}".format(i, purple, cost, cum_min[i-1], reset))
    else:
        print("{:<10d}{:<20.2f}{:<20.2f}".format(i, cost, cum_min[i-1]))

best_params = optimizer.max['params']

# ----------------------------
# Optimum parametrelerle 10 tekrar yapalım
results_list = []
for i in range(10):
    result = run(int(best_params['reorder_point']), int(best_params['order_size']))
    results_list.append(result['total_cost'])

print(f"\nOrtalama maliyet: {min(results_list)}")
print(f"Maliyet değişkenliği (Standart Sapma): {np.std(results_list)}")

# ----------------------------
# Ek Metriklerin Hesaplanması: RMSE, MAE, R-squared (R²)
y_true = np.array(results_list)
y_pred = np.full_like(y_true, np.mean(y_true))
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
print("\nEk Metrikler:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# Bitiş zamanını kaydedelim ve çalışma süresini hesaplayalım
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
