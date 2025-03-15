import numpy as np
import simpy
import itertools
import matplotlib.pyplot as plt
import csv

# -----------------------------------------------------------------------------
# SİMÜLASYON PARAMETRELERİ
# -----------------------------------------------------------------------------
MEAN_IAT = 0.1                      # Müşteri talepleri arasındaki ortalama süre (ay)
DEMAND_SIZES = [1, 2, 3, 4]         # Talep boyutu seçenekleri
DEMAND_PROB = [1/6, 1/3, 1/3, 1/6]  # Her talep boyutu için olasılık
START_INVENTORY = 60.0              # Başlangıç envanteri
COST_ORDER_SETUP = 32.0             # Sipariş başı sabit maliyet
COST_ORDER_PER_ITEM = 3.0           # Sipariş edilen her bir ürün için değişken maliyet
COST_BACKLOG_PER_ITEM = 5.0         # Backorder kalan her bir ürün için aylık ceza
COST_HOLDING_PER_ITEM = 1.0         # Envanterdeki her bir ürün için aylık stok tutma maliyeti
SIM_LENGTH = 120.0                  # Simülasyon süresi (ay)

# -----------------------------------------------------------------------------
# ENVANTER SİSTEMİ SINIFI
# -----------------------------------------------------------------------------
class InventorySystem:
    """
    Tek ürünlü envanter sistemi. 
    Sabit bir reorder point (ROP) ve sabit sipariş miktarı (order_size) politikası izlenir.
    Envanter belirli aralıklarla (burada 1 ayda bir) gözden geçirilir.
    """

    def __init__(self, env, reorder_point, order_size):
        # Parametreleri kaydet
        self.reorder_point = reorder_point
        self.order_size = order_size
        self.level = START_INVENTORY     # Güncel envanter seviyesi
        self.last_change = 0.0           # Envanterin en son değiştiği zaman
        self.ordering_cost = 0.0         # Toplam sipariş maliyeti
        self.shortage_cost = 0.0         # Toplam backorder (kıtlık) maliyeti
        self.holding_cost = 0.0          # Toplam stok tutma maliyeti
        self.history = [(0.0, START_INVENTORY)]  # (zaman, envanter seviyesi)

        # SimPy süreçlerini başlat
        env.process(self.review_inventory(env))
        env.process(self.demands(env))

    def place_order(self, env, units):
        """ 
        Belirli miktarda ürün sipariş eder ve tedarik süresi (lead time) sonunda envantere ekler.
        """
        # Sipariş maliyetini güncelle
        self.ordering_cost += (COST_ORDER_SETUP + units * COST_ORDER_PER_ITEM)
        # Rastgele bir tedarik süresi (lead time) hesapla (0.5 - 1.0 ay arası)
        lead_time = np.random.uniform(0.5, 1.0)
        yield env.timeout(lead_time)
        # Ürünler geldiğinde envanter seviyesini güncellemeden önce maliyeti hesapla
        self.update_cost(env)
        self.level += units
        self.last_change = env.now
        self.history.append((env.now, self.level))

    def review_inventory(self, env):
        """
        Belirli periyotlarda (1 ay) envanter seviyesini kontrol eder. 
        Eğer envanter, reorder point'in altındaysa sipariş verir.
        """
        while True:
            if self.level <= self.reorder_point:
                units = self.order_size + self.reorder_point - self.level
                env.process(self.place_order(env, units))
            yield env.timeout(1.0)

    def update_cost(self, env):
        """
        Her envanter hareketinde stok tutma ve kıtlık maliyetlerini günceller.
        """
        elapsed = env.now - self.last_change
        if self.level <= 0:
            # Kıtlık durumu: envanter 0 veya altında
            shortage_cost = abs(self.level) * COST_BACKLOG_PER_ITEM * elapsed
            self.shortage_cost += shortage_cost
        else:
            holding_cost = self.level * COST_HOLDING_PER_ITEM * elapsed
            self.holding_cost += holding_cost

    def demands(self, env):
        """
        Müşteri taleplerini rastgele zaman aralıklarında (exponential dağılım) üretir.
        """
        while True:
            iat = np.random.exponential(MEAN_IAT)
            yield env.timeout(iat)
            size = np.random.choice(DEMAND_SIZES, p=DEMAND_PROB)
            self.update_cost(env)
            self.level -= size
            self.last_change = env.now
            self.history.append((env.now, self.level))

# -----------------------------------------------------------------------------
# TEK BİR SİMÜLASYON ÇALIŞTIRMA FONKSİYONU
# -----------------------------------------------------------------------------
def run(reorder_point: float, order_size: float, display_chart=False):
    """
    Belirtilen reorder point ve order size ile tek bir simülasyon çalıştırır.
    Simülasyon sonunda ortalama toplam maliyet ve bileşen maliyetleri döndürülür.
    """
    if SIM_LENGTH <= 0:
        raise ValueError("Simulation length must be greater than zero")
    if order_size < 0:
        raise ValueError("Order size must be greater than zero")  
    
    env = simpy.Environment()
    inv = InventorySystem(env, reorder_point, order_size)
    env.run(SIM_LENGTH)

    avg_total_cost = (inv.ordering_cost + inv.holding_cost + inv.shortage_cost) / SIM_LENGTH
    avg_ordering_cost = inv.ordering_cost / SIM_LENGTH
    avg_holding_cost = inv.holding_cost / SIM_LENGTH
    avg_shortage_cost = inv.shortage_cost / SIM_LENGTH

    results = {
        'reorder_point': reorder_point,
        'order_size': order_size,
        'total_cost': round(avg_total_cost, 1), 
        'ordering_cost': round(avg_ordering_cost, 1),
        'holding_cost': round(avg_holding_cost, 1), 
        'shortage_cost': round(avg_shortage_cost, 1)
    }

    if display_chart:
        step_graph(inv)
    
    return results

# -----------------------------------------------------------------------------
# ENVANTER SEVİYESİ GRAFİĞİ
# -----------------------------------------------------------------------------
def step_graph(inventory):
    """ Envanter seviyesinin zamana göre değişimini adım grafiği olarak gösterir. """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(which='major', alpha=0.4)
    x_val = [x[0] for x in inventory.history]
    y_val = [x[1] for x in inventory.history]
    plt.step(x_val, y_val, where='post', label='Units in inventory')
    plt.axhline(y=0, color='red', linestyle='-', label='Shortage threshold')
    plt.axhline(y=inventory.reorder_point, color='green', linestyle='--', label='Reorder point')
    plt.xlabel('Months')
    plt.ylabel('Units in inventory')
    plt.title(f'Simulation output (ROP={inventory.reorder_point}, Q={inventory.order_size})')
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# TEKRARLI SİMÜLASYON (EXPERIMENTS) FONKSİYONU
# -----------------------------------------------------------------------------
def run_experiments(reorder_points, order_sizes, num_rep):
    """
    Belirtilen reorder point ve order size dizileri için num_rep kez tekrarlı simülasyon çalıştırır.
    Sonuçları, her deney için bir sözlük içeren liste olarak döndürür.
    """
    if num_rep <= 0:
        raise ValueError('Number of replications must be greater than zero')
    
    results = []
    total_iter = len(reorder_points) * len(order_sizes) * num_rep
    iter_count = 0

    for rp, q in itertools.product(reorder_points, order_sizes):
        for _ in range(num_rep):
            iter_count += 1
            if iter_count % 100 == 0:
                print('Iteration', iter_count, 'of', total_iter)
            sim_result = run(rp, q)
            results.append(sim_result)
    return results

# -----------------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU: SONUÇLARI CSV'YE YAZDIRMA
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Deney parametreleri: 
    # Reorder point 0'dan 100'e kadar 5'er artış, order size 5'ten 50'ye kadar 5'er artış
    reorder_points = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
    order_sizes = list(range(5, 55, 5))        # 5, 10, 15, ..., 50
    num_rep = 10  # Her kombinasyon için 10 tekrar

    # Tüm deneyleri çalıştır
    print("Deneyler başlatılıyor...")
    all_results = run_experiments(reorder_points, order_sizes, num_rep)
    print("Deneyler tamamlandı. Toplam sonuç sayısı:", len(all_results))

    # Sonuçları CSV dosyasına yazdırma
    csv_filename = "results.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # CSV başlığı: ilk sütun index, sonrasında parametreler ve maliyetler
        writer.writerow(["index", "reorder_point", "order_size", "total_cost", "ordering_cost", "holding_cost", "shortage_cost"])
        for idx, r in enumerate(all_results):
            writer.writerow([
                idx,
                r['reorder_point'],
                r['order_size'],
                r['total_cost'],
                r['ordering_cost'],
                r['holding_cost'],
                r['shortage_cost']
            ])
    print(f"Sonuçlar '{csv_filename}' dosyasına yazdırıldı.")
