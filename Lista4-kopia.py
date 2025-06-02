import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Stałe fizyczne ---
G = 6.67430e-11         # Stała grawitacyjna [m^3 kg^-1 s^-2]
M = 1.989e30            # Masa Słońca [kg]

# --- Parametry początkowe: zbyt mała prędkość ---
x0 = 1.496e11           # Odległość początkowa od Słońca [m]
y0 = 0
vx0 = 0
vy0 = 20000             # Zbyt mała prędkość orbitalna [m/s] (normalnie ~29780)

# --- Funkcja pochodnych ---
def dydt(t, y):
    x, y_pos, vx, vy = y
    r = np.sqrt(x**2 + y_pos**2)
    ax = -G * M * x / r**3
    ay = -G * M * y_pos / r**3
    return [vx, vy, ax, ay]

# --- Czas symulacji ---
T = 1.5e7  # [s] skrócony, bo ciało może szybko spaść
dt = 10000
t_eval = np.arange(0, T, dt)
t_span = (0, T)

# --- Rozwiązanie numeryczne ---
sol = solve_ivp(dydt, t_span, [x0, y0, vx0, vy0], t_eval=t_eval, method='RK45')

# --- Wykres trajektorii ---
plt.figure()
plt.plot(sol.y[0], sol.y[1], label='Trajektoria planety', color='blue')
plt.scatter([0], [0], color='orange', label='Słońce', s=100)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Zbyt mała prędkość początkowa – planeta spada na Słońce')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



