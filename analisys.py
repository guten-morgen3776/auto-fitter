import numpy as np
from scipy.optimize import curve_fit

class BaseModel:
    def __init__(self):
        self.params = None
        self.k = 0
        self.x_data = None
        self.y_data = None
    
    def fit(self, x, y):
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        self.params, _ = curve_fit(self.func, self.x_data, self.y_data)

    def get_aic(self):
        y_pred = self.func(self.x_data, *self.params)
        rss = np.sum((self.y_data - y_pred) ** 2)
        n = len(self.y_data)
        aic = n * np.log(rss / n) + 2 * self.k
        return aic
    
    def get_equation(self):
        return ''
    
class LinearModel(BaseModel):
    def __init__(self):
        super().__init__() #親クラスのinitを呼ぶ
        self.k = 2
    
    def func(self, x, a, b):
        return a * x + b
    
    def get_equation(self):
        a, b = self.params
        return f'y = {a: 3f}x + {b: 3f}'

class QuadraticModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 3
    
    def func(self, x, a, b, c):
        return a * x ** 2 + b * x + c
    
    def get_equation(self):
        a, b, c = self.params
        return f'y = {a: 3f}x^2 + {b: 3f}x + {c: 3f}'
    
class ExponentialModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 3
    
    def func(self, x, a, b, c):
        return a * np.exp(b * x) + c
    
    def get_equation(self):
        a, b, c = self.params
        return f"y = {a:.3f}e^{{{b:.3f}x}} + {c:.3f}"