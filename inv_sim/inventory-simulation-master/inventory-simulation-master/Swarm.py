import time
import numpy as np
import matplotlib.pyplot as plt
from model import run  # Simülasyon fonksiyonunu içe aktarıyoruz
import pyswarms as ps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Başlangıç zamanını kaydedelim
start_time = time.time()

# ----------------------------
# PSO Objective Fonksiyonu
# ----------------------------
def pso_objective_function(x):
    """
    x: Array şeklinde (n_particles, 2) aday parametreler.
       x[:,0] -> reorder_point, x[:,1] -> order_size.
    Her aday için 5 simülasyon çalıştırılarak ortalama total_cost hesaplanır.
    PSO, minimize etmeye çalışır.
    """
    n_particles = x.shape[0]
    costs = np.zeros(n_particles)
    for i in range(n_particles):
        candidate_costs = []
        for _ in range(5):
            # Parametreleri tamsayıya çevirerek simülasyonu çalıştırıyoruz.
            result = run(int(x[i, 0]), int(x[i, 1]))
            candidate_costs.append(result['total_cost'])
        costs[i] = np.mean(candidate_costs)
    return costs

# ----------------------------
# Karar Değişkenleri Aralıkları
# ----------------------------
# Lower bounds: reorder_point = 0, order_size = 5
# Upper bounds: reorder_point = 100, order_size = 50
bounds = (np.array([0, 5]), np.array([100, 50]))

# PSO ayarları
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# ----------------------------
# PSO Optimizasyonu
# ----------------------------
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=bounds)
best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=100)

print("\nPSO En İyi Parametreler (Best Position):", best_pos)
print("PSO En İyi Total Cost:", best_cost)

# ----------------------------
# İterasyon Tablosu (Kümülatif Minimum) Yazdırma
# ----------------------------
# optimizer.cost_history: her iterasyonda elde edilen en iyi cost değerleri
cost_history = optimizer.cost_history
cum_min = []
current_min = float('inf')

# Renk kodları: yeni en iyi bulunursa mor renkle vurgulayalım.
purple = "\033[95m"
reset = "\033[0m"

print("\nIteration   Cost               Cumulative Min")
for idx, cost in enumerate(cost_history, start=1):
    new_best = False
    if cost < current_min:
        current_min = cost
        new_best = True
    if new_best:
        print("{:<12d}{}{:<18.2f}{}{}{:<18.2f}{}".format(idx, purple, cost, reset, purple, current_min, reset))
    else:
        print("{:<12d}{:<18.2f}{:<18.2f}".format(idx, cost, current_min))
    cum_min.append(current_min)

# ----------------------------
# Kümülatif En İyi (Minimum) Cost Grafiğini Çizdirme
# ----------------------------
iterations = np.arange(1, len(cum_min) + 1)
plt.figure(figsize=(10, 6))
plt.plot(iterations, cum_min, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Minimum Total Cost')
plt.title('Cumulative Minimum Total Cost over PSO Iterations')
plt.grid(True)
plt.show()

# ----------------------------
# Doğrulama (Validation)
# ----------------------------
results = []
for i in range(10):
    res = run(int(best_pos[0]), int(best_pos[1]))
    results.append(res['total_cost'])

print(f"\nDoğrulama: Ortalama Total Cost: {np.mean(results):.2f}")
print(f"Total Cost Standart Sapması: {np.std(results):.2f}")

# ----------------------------
# Ek Metrikler: RMSE, MAE, R-squared (R²)
# ----------------------------
# 10 tekrarın ortalamasını "tahmin" (y_pred) olarak kabul ediyoruz,
# ve her tekrarı "gerçek" (y_true) olarak.
y_true = np.array(results)
y_pred = np.full_like(y_true, np.mean(y_true))
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"\nEk Metrikler:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# ----------------------------
# Hassasiyet (Sensitivity) Analizi
# ----------------------------
def sensitivity_analysis(best_params):
    for delta in [-10, 0, 10]:  # -%10, 0, +%10 değişiklik
        rop = int(best_params[0] * (1 + delta / 100))
        order_size = int(best_params[1] * (1 + delta / 100))
        res = run(rop, order_size)
        print(f"ROP: {rop}, Order Size: {order_size} → Total Cost: {res['total_cost']}")

print("\nHassasiyet Analizi Sonuçları:")
sensitivity_analysis(best_pos)

# ----------------------------
# Grafik: Maliyet Bileşenleri
# ----------------------------
def plot_costs(sim_result):
    plt.figure(figsize=(8, 5))
    plt.bar(['Ordering', 'Holding', 'Shortage'], 
            [sim_result['ordering_cost'], sim_result['holding_cost'], sim_result['shortage_cost']])
    plt.title('Cost Components')
    plt.ylabel('Cost')
    plt.show()

plot_costs(run(int(best_pos[0]), int(best_pos[1])))

# ----------------------------
# Çalışma Süresi (Timer)
# ----------------------------
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
