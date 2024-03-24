from pulp import LpProblem, LpMaximize, LpVariable, value
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from tabulate import tabulate

class MonteCarloIntegration:
    def __init__(self, func, a, b, samples=10000):
        self.func = func
        self.a = a
        self.b = b
        self.samples = samples

    def integrate(self):
        random_samples = np.random.uniform(self.a, self.b, self.samples)
        func_values = self.func(random_samples)
        area = (self.b - self.a) * np.mean(func_values)
        return area

    def compare_with_quad(self):
        mc_result = self.integrate()
        quad_result, _ = spi.quad(self.func, self.a, self.b)
        results = [
            ["Method", "Result"],
            ["Monte Carlo", mc_result],
            ["Quad", quad_result],
        ]
        print(tabulate(results, headers="firstrow", tablefmt="pipe"))
        self.visualize()

    def visualize(self):
        x = np.linspace(self.a - 1, self.b + 1, 400)
        y = self.func(x)
        plt.plot(x, y, "r", linewidth=2)
        plt.fill_between(x, y, color="gray", alpha=0.3)
        plt.title(f"Integration of f(x) from {self.a} to {self.b}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()


def main():
    # Виконання та візуалізація Монте-Карло інтеграції
    def f(x):
        return x**2

    mc_integration = MonteCarloIntegration(f, 0, 2)
    mc_integration.compare_with_quad()


if __name__ == "__main__":
    main()