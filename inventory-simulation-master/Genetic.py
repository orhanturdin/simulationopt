import time
from model import run  # Simülasyon fonksiyonunu içe aktarıyoruz
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Eklenen: sklearn metriklerini içe aktarıyoruz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Başlangıç zamanını kaydedelim
start_time = time.time()

# Global değişkenler: iterasyon sayacı, en iyi cost takibi, renk kodları
iteration_count = 0
best_cost_so_far = float('inf')
purple = "\033[95m"
reset = "\033[0m"

# GA objective fonksiyonu: Verilen parametreler için simülasyonu çalıştırıp total_cost döndürür.
def ga_objective_function(x):
    # x[0]: reorder_point, x[1]: order_size
    result = run(int(x[0]), int(x[1]))
    return result['total_cost']

# Callback fonksiyonu: Her iterasyonda çağrılır ve tablo şeklinde çıktı verir.
def de_callback(xk, convergence):
    global iteration_count, best_cost_so_far
    iteration_count += 1
    # Şu anki aday parametreler için objective fonksiyonunu hesaplayalım
    cost = ga_objective_function(xk)

    # Kümülatif minimum kontrolü
    new_best = False
    if cost < best_cost_so_far:
        best_cost_so_far = cost
        new_best = True

    # Tablo satırı olarak yazdır: Iteration, Reorder Point, Order Size, Total Cost, Cumulative Min
    # Eğer yeni en iyi değer bulunmuşsa (new_best=True), cost ve cumulative min'i mor renkte göster
    if new_best:
        print("{:<10d}{:<20.2f}{:<20.2f}{}{:<20.2f}{}{}{:<20.2f}{}".format(
            iteration_count, xk[0], xk[1],
            purple, cost, reset,
            purple, best_cost_so_far, reset
        ))
    else:
        print("{:<10d}{:<20.2f}{:<20.2f}{:<20.2f}{:<20.2f}".format(
            iteration_count, xk[0], xk[1],
            cost, best_cost_so_far
        ))
    # False döndürmek, optimizasyonun devam etmesini sağlar.
    return False

# Tablo başlığı yazdırılır.
print("{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}".format(
    "Iteration", "Reorder Point", "Order Size", "Total Cost", "Cumulative Min"
))

# Differential Evolution ile optimizasyon (Genetic Algorithm)
result = differential_evolution(
    func=ga_objective_function,
    bounds=[(0, 100), (5, 50)],  # Parametre aralıkları
    strategy='best1bin',
    maxiter=100,
    callback=de_callback
)

# GA sonucunda elde edilen en iyi parametreleri sözlük olarak saklayalım.
best_params = {'reorder_point': result.x[0], 'order_size': result.x[1]}
print("\nGA En İyi Çözüm Parametreleri:", result.x)

# GA en iyi parametrelerle simülasyonu çalıştırıp total_cost'u alıyoruz.
ga_best_result = run(int(result.x[0]), int(result.x[1]))
print("GA En İyi Çözüm Total Cost:", ga_best_result['total_cost'])

# ----------------------- Doğrulama (Validation) -----------------------
results = []
for i in range(10):
    res = run(int(result.x[0]), int(result.x[1]))
    results.append(res['total_cost'])

print(f"\nMin maliyet: {min(results)}")
print(f"Maliyet değişkenliği (Standart Sapma): {np.std(results)}")

# Eklenen: RMSE, MAE, R-squared (R²) hesaplaması
# Basitçe, 10 tekrarın ortalamasını "tahmin" (y_pred) olarak kabul ediyoruz
# ve her tekrarı "gerçek" (y_true) kabul ediyoruz.
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

# ----------------------- Hassasiyet (Sensitivity) Analizi -----------------------
def sensitivity_analysis(best_params):
    for delta in [-10, 0, 10]:  # -%10, 0, +%10 değişiklik
        rop = int(best_params['reorder_point'] * (1 + delta / 100))
        order_size = int(best_params['order_size'] * (1 + delta / 100))
        res = run(rop, order_size)
        print(f"ROP: {rop}, Order Size: {order_size} → Toplam Maliyet: {res['total_cost']}")

print("\nHassasiyet Analizi Sonuçları:")
sensitivity_analysis(best_params)

# ----------------------- Grafik ve Raporlama -----------------------
def plot_costs(sim_result):
    plt.figure(figsize=(8, 5))
    plt.bar(['Ordering', 'Holding', 'Shortage'], 
            [sim_result['ordering_cost'], sim_result['holding_cost'], sim_result['shortage_cost']])
    plt.title('Maliyet Bileşenleri')
    plt.ylabel('Maliyet')
    plt.show()

plot_costs(run(int(result.x[0]), int(result.x[1])))

# Bitiş zamanını kaydedelim ve çalışma süresini hesaplayalım
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time:.2f} seconds")
