import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# === Funkcja opisująca siłę centralną (grawitacyjną) ===
def orbital_rhs(t, y, k):
    x, vx, y_, vy = y
    r = np.sqrt(x**2 + y_**2)
    ax = -k * x / r**3
    ay = -k * y_ / r**3
    return [vx, ax, vy, ay]

# === Moment pędu na jednostkę masy ===
def compute_h(x0, vx0, y0, vy0):
    return x0 * vy0 - y0 * vx0

# === Symulacja toru z danymi fizycznymi ===
def run_simulation_phys(G, M, y0, t_span, label_suffix="", num_points=500):
    k = G * M
    h = compute_h(*y0)
    B = k / h**2
    A = 1  # domyślna wartość do porównania (można doprecyzować)

    theta_vals = np.linspace(0, 2 * np.pi, num_points)
    r_exact = 1 / (A * np.cos(theta_vals) + B)
    x_exact = r_exact * np.cos(theta_vals)
    y_exact = r_exact * np.sin(theta_vals)

    t_eval = np.linspace(*t_span, num_points)
    sol = solve_ivp(orbital_rhs, t_span, y0, args=(k,), t_eval=t_eval, rtol=1e-9)

    x_num = sol.y[0]
    y_num = sol.y[2]
    r_numeric = np.sqrt(x_num**2 + y_num**2)
    theta_numeric = np.arctan2(y_num, x_num)
    sort_idx = np.argsort(theta_numeric)
    theta_sorted = theta_numeric[sort_idx]
    r_sorted_numeric = r_numeric[sort_idx]
    r_interp_exact = 1 / (A * np.cos(theta_sorted) + B)

    abs_error = np.abs(r_sorted_numeric - r_interp_exact)
    sq_error = (r_sorted_numeric - r_interp_exact)**2
    mae = np.mean(abs_error)
    mse = np.mean(sq_error)

    # === Wykresy ===
    plt.figure()
    plt.polar(theta_sorted, r_sorted_numeric, label='Numeryczne')
    plt.polar(theta_sorted, r_interp_exact, label='Dokładne', linestyle='--')
    plt.title(f"Trajektoria r(θ) {label_suffix}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(theta_sorted, abs_error)
    plt.title(f"Błąd bezwzględny  {label_suffix}")
    plt.xlabel("θ [rad]")
    plt.ylabel("Błąd")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(theta_sorted, sq_error)
    plt.title(f"Błąd kwadratowy  {label_suffix}")
    plt.xlabel("θ [rad]")
    plt.ylabel("Błąd")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(sol.t, r_numeric)
    plt.title(f"Odległość r(t) – numerycznie {label_suffix}")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_num, y_num, sol.t, label=f'Tor 3D {label_suffix}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(f"Trajektoria 3D: x(t), y(t), t {label_suffix}")
    plt.legend()
    plt.show()

    return mae, mse

# === Lista zestawów do symulacji ===
params_phys = [
    (6.67430e-11, 1.989e30, [1.496e11, 0, 0, 29780], (0, 3.154e7), "Ziemia – fizyczne dane"),
    (1, 1, [1, 0, 0, 1], (0, 10), "Jednostki uproszczone"),
    (1, 2, [1, 0, 0, 1], (0, 10), "Zwiększona masa M=2"),
    (1, 1, [1.2, 0, 0, 0.8], (0, 10), "Zmienione położenie i prędkość")
]

results_phys = []
for G, M, y0, t_span, label in params_phys:
    mae, mse = run_simulation_phys(G, M, y0, t_span, label)
    results_phys.append((label, mae, mse))

# === Tabela wyników ===
df_phys = pd.DataFrame(results_phys, columns=["Opis", "MAE", "MSE"])
print("\nPorównanie wyników dla różnych zestawów fizycznych parametrów:")
print(df_phys)

