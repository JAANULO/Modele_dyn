#lista4

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import sympy as sp


# === Równania ruchu ===
def orbital_rhs(t, y, k):
    x, vx, y_, vy = y
    r = np.sqrt(x**2 + y_**2)
    ax = -k * x / r**3
    ay = -k * y_ / r**3
    return [vx, ax, vy, ay]

def compute_h(x0, vx0, y0, vy0):
    return x0 * vy0 - y0 * vx0

def compute_e(x0, vx0, y0, vy0, G, M):
    r0 = np.sqrt(x0**2 + y0**2)
    v0 = np.sqrt(vx0**2 + vy0**2)
    h = compute_h(x0, vx0, y0, vy0)
    mu = G * M
    e = np.sqrt(1 + (2 * (0.5 * v0**2 - mu / r0) * h**2) / mu**2)
    return e

# === Symboliczne rozwiązanie r(θ) z sympy ===
def symbolic_r_theta(h_val, G_val, M_val, e_val):
    theta = sp.symbols('theta')
    mu = G_val * M_val
    p = h_val**2 / mu
    r_exact_expr = p / (1 + e_val * sp.cos(theta))
    return r_exact_expr

# === Główna funkcja do rysowania r(θ) ===
def plot_theta_trajectories_by_group(params_phys, num_points=500):
    results_phys = []

    for i in range(0, len(params_phys), 3):
        group = params_phys[i:i+3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        axs = axs if isinstance(axs, np.ndarray) else [axs]

        for ax, (G, M, y0, t_span, label) in zip(axs, group):
            k = G * M
            h = compute_h(*y0)
            e = compute_e(*y0, G, M)
            p = h**2 / k

            t_eval = np.linspace(*t_span, num_points)
            sol = solve_ivp(orbital_rhs, t_span, y0, args=(k,), t_eval=t_eval, rtol=1e-9)

            x_num = sol.y[0]
            y_num = sol.y[2]
            r_numeric = np.sqrt(x_num**2 + y_num**2)
            theta_numeric = np.arctan2(y_num, x_num)

            # sortujemy dane po kącie
            sort_idx = np.argsort(theta_numeric)
            theta_sorted = theta_numeric[sort_idx]
            r_sorted_numeric = r_numeric[sort_idx]

            # dokładne rozwiązanie z sympy
            r_exact_expr = symbolic_r_theta(h, G, M, e)
            r_exact_func = sp.lambdify(sp.symbols('theta'), r_exact_expr, modules=['numpy'])
            r_exact_sorted = r_exact_func(theta_sorted)

            # błędy
            abs_error = np.abs(r_sorted_numeric - r_exact_sorted)
            sq_error = (r_sorted_numeric - r_exact_sorted)**2
            mae = np.mean(abs_error)
            mse = np.mean(sq_error)
            results_phys.append((label, mae, mse))

            # wykres r(θ)
            ax.plot(theta_sorted, r_sorted_numeric, label="Numeryczne", color='black')
            ax.plot(theta_sorted, r_exact_sorted, 'r--', label="Dokładne (SymPy)")
            ax.set_title(label)
            ax.set_xlabel("θ [rad]")
            ax.grid(True)
            if ax == axs[0]:
                ax.set_ylabel("r")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2)
        #fig.suptitle("Porównanie r(θ): Numeryczne vs Dokładne (SymPy)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

    return results_phys

# === 3D wykresy trajektorii po 3 przypadki ===
def plot_3d_trajectories_by_group(params_phys, num_points=500):
    for i in range(0, len(params_phys), 3):
        group = params_phys[i:i+3]
        fig = plt.figure(figsize=(18, 6))

        for j, (G, M, y0, t_span, label) in enumerate(group):
            k = G * M
            t_eval = np.linspace(*t_span, num_points)
            sol = solve_ivp(orbital_rhs, t_span, y0, args=(k,), t_eval=t_eval, rtol=1e-9)
            x_num = sol.y[0]
            y_num = sol.y[2]
            t_num = sol.t

            ax = fig.add_subplot(1, 3, j + 1, projection='3d')
            ax.plot(x_num, y_num, t_num, label=label)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("t")
            ax.set_title(label)
            ax.legend()
            ax.grid(True)

        fig.suptitle("Trajektorie 3D: x(t), y(t), t", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# === Zestaw przypadków fizycznych ===
params_phys = [
    (6.67430e-11, 1.989e30, [1.496e11, 0, 0, 29780], (0, 3.154e7), "Ziemia – fizyczne dane"),
    (6.67430e-11, 1.989e30, [4.6e10, 0, 0, 58980], (0, 7.6e6), "Merkury – fizyczne dane"),
    (6.67430e-11, 1.989e30, [7.5e10, 0, 0, 47000], (0, 1.0e7), "Silnie eliptyczna orbita"),
    (6.67430e-11, 1.989e30, [1.496e11, 0, 0, 15000], (0, 3.154e7), "Zbyt wolna prędkość – spiralny spadek"),
    (6.67430e-11, 1.989e30, [1.496e11, 0, 0, 60000], (0, 1.0e7), "Zbyt duża prędkość – ucieczka"),
    (6.67430e-11, 5 * 1.989e30, [1.496e11, 0, 0, 29780], (0, 1.5e7), "Masa Słońca ×5 – silniejsze pole"),
]

# === Uruchomienie wszystkiego ===
results_phys = plot_theta_trajectories_by_group(params_phys)
df_phys = pd.DataFrame(results_phys, columns=["Opis", "MAE", "MSE"])
print("\nPorównanie wyników dla różnych zestawów fizycznych parametrów:")
print(df_phys)

plot_3d_trajectories_by_group(params_phys)

