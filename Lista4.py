# lista4

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Parametry bazowe
G = 1
M = 1
k = G * M

# Funkcja RHS dla solve_ivp
def orbital_rhs(t, y, k):
    x, vx, y_, vy = y
    r = np.sqrt(x**2 + y_**2)
    ax = -k * x / r**3
    ay = -k * y_ / r**3
    return [vx, ax, vy, ay]

# Funkcja pomocnicza do analizy jednej symulacji
def run_simulation(A, h, y0, label_suffix="", num_points=500):
    B = k / h**2
    theta_vals = np.linspace(0, 2 * np.pi, num_points)
    r_exact = 1 / (A * np.cos(theta_vals) + B)
    x_exact = r_exact * np.cos(theta_vals)
    y_exact = r_exact * np.sin(theta_vals)

    # Rozwiązanie numeryczne (SciPy)
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, num_points)
    sol = solve_ivp(orbital_rhs, t_span, y0, args=(k,), t_eval=t_eval, rtol=1e-9)

    # Przeliczenie do r(t) i θ(t)
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

    # Wykresy
    plt.figure()
    plt.polar(theta_sorted, r_sorted_numeric, label='Numeryczne')
    plt.polar(theta_sorted, r_interp_exact, label='Dokładne', linestyle='--')
    plt.title(f"Trajektoria r(θ) {label_suffix}")
    plt.legend()

    plt.figure()
    plt.plot(theta_sorted, abs_error)
    plt.title(f"Błąd bezwzględny |r_num - r_exact| {label_suffix}")
    plt.xlabel("θ [rad]")
    plt.ylabel("Błąd")
    plt.grid(True)

    plt.figure()
    plt.plot(theta_sorted, sq_error)
    plt.title(f"Błąd kwadratowy (r_num - r_exact)^2 {label_suffix}")
    plt.xlabel("θ [rad]")
    plt.ylabel("Błąd")
    plt.grid(True)

    plt.figure()
    plt.plot(sol.t, r_numeric)
    plt.title(f"Odległość r(t) – numerycznie {label_suffix}")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.grid(True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_num, y_num, sol.t, label=f'Tor 3D {label_suffix}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(f"Trajektoria 3D: x(t), y(t), t {label_suffix}")
    plt.legend()


    return mae, mse

# Symulacje dla różnych zestawów parametrów
params = [
    (0.5, 1.0, [1, 0, 0, 1], "(A=0.5, h=1.0)"),
    (0.8, 1.2, [1.1, 0, 0, 0.9], "(A=0.8, h=1.2)"),
    (0.3, 0.9, [0.9, 0, 0, 1.1], "(A=0.3, h=0.9)")
]

results = []
for A, h, y0, label in params:
    mae, mse = run_simulation(A, h, y0, label)
    results.append((label, mae, mse))

# Tabela wyników
df = pd.DataFrame(results, columns=["Parametry", "MAE", "MSE"])
print("\nPorównanie wyników dla różnych parametrów:")
print(df)

plt.show()

