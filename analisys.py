import numpy as np
from scipy.optimize import curve_fit

class BaseModel:
    def __init__(self):
        self.params = None
        self.cov = None
        self.errors = None
        self.k = 0
        self.x_data = None
        self.y_data = None
    
    def fit(self, x, y):
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        popt, pcov = curve_fit(self.func, self.x_data, self.y_data, maxfev=5000)
        self.params = popt
        self.errors = np.sqrt(np.diag(pcov))
        self.cov = pcov
    
    def get_confidence_interval(self, x_domain, n_samples=1000, alpha=0.05):
        sampled_params = np.random.multivariate_normal(self.params, self.cov, n_samples)
        sampled_y = []
        for p in sampled_params:
            sampled_y.append(self.func(x_domain, *p))
        sampled_y = np.array(sampled_y)
        lower_bound = np.percentile(sampled_y, 100 * (alpha / 2), axis=0)
        upper_bound = np.percentile(sampled_y, 100 * (1 - alpha / 2), axis=0)
        return lower_bound, upper_bound
    
    def get_aicc(self):
        y_pred = self.func(self.x_data, *self.params)
        rss = np.sum((self.y_data - y_pred) ** 2)
        n = len(self.y_data)
        k = self.k
        aic = n * np.log(rss / n) + 2 * k
        correction_term = (2 * k * (k + 1)) / (n - k - 1)
        if n - k - 1 <= 0:
            return np.inf
        return aic + correction_term
    
    def get_equation(self):
        return ''
    
class LinearModel(BaseModel):
    def __init__(self):
        super().__init__() 
        self.k = 2
    
    def func(self, x, a, b):
        return a * x + b
    
    def get_equation(self):
        a, b = self.params
        ea, eb = self.errors
        return f'y = ({a: 3f} ± {ea:.3f})x + ({b: 3f} ± {eb:.3f})'

class QuadraticModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 3
    
    def func(self, x, a, b, c):
        return a * x ** 2 + b * x + c
    
    def get_equation(self):
        a, b, c = self.params
        ea, eb, ec = self.errors
        return f'y = ({a:.3f} ± {ea:.3f})x^2 + ({b:.3f} ± {eb:.3f})x + ({c:.3f} ± {ec:.3f})'
    
class ExponentialModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 3
    
    def func(self, x, a, b, c):
        return a * np.exp(b * x) + c
    
    def get_equation(self):
        a, b, c = self.params
        ea, eb, ec = self.errors
        return f"y = ({a:.3f} ± {ea:.3f})e^{{({b:.3f} ± {eb:.3f})x}} + ({c:.3f} ± {ec:.3f})"
    
class LogarithmicModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 2
    
    def func(self, x, a, b):
        return a * np.log(x) + b
    
    def get_equation(self):
        if self.params is None: return 'Fitting Failed'
        a, b = self.params
        ea, eb = self.errors
        return f'y = ({a:.3f} ± {ea:.3f})ln(x) + ({b:.3f} ± {eb:.3f})'
    
class SigmoidModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 3
    
    def func(self, x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))
    
    def get_equation(self):
        a, b, c = self.params
        return f'y = {a:.3f} / (1 + e^({-b:.3f}(x - {c:.3f})))'

class MichaelisMentenModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.k = 2 # a: Vmax, b: Km

    def func(self, x, a, b):
        return (a * x) / (b + x)

    def get_equation(self):
        a, b = self.params
        ea, eb = self.errors
        # a=Vmax, b=Km
        return f"y = ({a:.3f} ± {ea:.3f})x / (({b:.3f} ± {eb:.3f}) + x)"