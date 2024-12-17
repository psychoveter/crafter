import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Генерируем данные
np.random.seed(42)
true_slope = 2.5
true_intercept = 1.0
n_samples = 100
x = np.linspace(0, 10, n_samples)
y = true_slope * x + true_intercept + np.random.normal(0, 1, size=n_samples)

# Байесовская модель
with pm.Model() as model:
    # Приоритеты (Prior)
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Линейная модель
    y_obs = pm.Normal("y_obs", mu=slope * x + intercept, sigma=sigma, observed=y)

    # Запускаем MCMC
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# Визуализация результата
import arviz as az

az.plot_trace(trace)
plt.show()

# Сводная статистика
print(az.summary(trace, round_to=2))