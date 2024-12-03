import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    result = 0
    
    for i in range(n):
        term = y_points[i]
        
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
                
        result += term
        
    return result

def linear_regression(x, y):
    n = len(x)
    
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - m * np.sum(x)) / n
    
    return m, b

def power_law_regression(x, y):
    # Логарифмуємо дані
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Лінійна регресія в логарифмічній системі координат
    n = len(log_x)
    b = (n * np.sum(log_x * log_y) - np.sum(log_x) * np.sum(log_y)) / (n * np.sum(log_x**2) - np.sum(log_x)**2)
    log_a = (np.sum(log_y) - b * np.sum(log_x)) / n
    
    # Перетворюємо назад
    a = np.exp(log_a)
    
    return a, b

class InterpolationApp:
    def __init__(self, master):
        self.master = master
        master.title("Інтерполяція та Регресія")
        master.geometry("800x600")

        self.x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.y_data = np.array([56.9, 67.3, 81.6, 201, 240, 474, 490, 518])

        # Створення фреймів для введення даних та керування
        self.input_frame = ttk.Frame(master, padding="10")
        self.input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Мітки та поля введення для x та y
        ttk.Label(self.input_frame, text="X точки:").grid(row=0, column=0)
        self.x_entry = ttk.Entry(self.input_frame, width=50)
        self.x_entry.grid(row=0, column=1)
        self.x_entry.insert(0, " ".join(map(str, self.x_data)))

        ttk.Label(self.input_frame, text="Y точки:").grid(row=1, column=0)
        self.y_entry = ttk.Entry(self.input_frame, width=50)
        self.y_entry.grid(row=1, column=1)
        self.y_entry.insert(0, " ".join(map(str, self.y_data)))

        # Кнопки для побудови графіків
        ttk.Button(self.input_frame, text="Лагранж", command=self.plot_lagrange).grid(row=2, column=0)
        ttk.Button(self.input_frame, text="Лінійна Регресія", command=self.plot_linear_regression).grid(row=2, column=1)
        ttk.Button(self.input_frame, text="Степенева Регресія", command=self.plot_power_law_regression).grid(row=2, column=2)

    def _get_input_data(self):
        try:
            x_input = list(map(float, self.x_entry.get().split()))
            y_input = list(map(float, self.y_entry.get().split()))
            return np.array(x_input), np.array(y_input)
        except ValueError:
            tk.messagebox.showerror("Помилка", "Некоректний формат введення даних")
            return None, None

    def plot_lagrange(self):
        x, y = self._get_input_data()
        if x is None or y is None:
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='red', label='Вхідні точки')
        
        x_interp = np.linspace(min(x), max(x), 200)
        y_interp = [lagrange_interpolation(x, y, xi) for xi in x_interp]
        
        plt.plot(x_interp, y_interp, label='Інтерполяція Лагранжа')
        plt.title('Інтерполяція Лагранжа')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_linear_regression(self):
        x, y = self._get_input_data()
        if x is None or y is None:
            return

        m, b = linear_regression(x, y)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='red', label='Вхідні точки')
        
        x_line = np.linspace(min(x), max(x), 200)
        y_line = m * x_line + b
        
        plt.plot(x_line, y_line, label=f'Лінійна Регресія: y = {m:.2f}x + {b:.2f}')
        plt.title('Лінійна Регресія')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_power_law_regression(self):
        x, y = self._get_input_data()
        if x is None or y is None:
            return

        a, b = power_law_regression(x, y)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='red', label='Вхідні точки')
        
        x_line = np.linspace(min(x), max(x), 200)
        y_line = a * x_line ** b
        
        plt.plot(x_line, y_line, label=f'Степенева Регресія: y = {a:.2f} * x^{b:.2f}')
        plt.title('Степенева Регресія')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()