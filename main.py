import numpy as np
from analisys import LinearModel

x = [1, 2, 3, 4, 5]
y = [7.1, 8.9, 11.2, 12.8, 15.1]

model = LinearModel()
model.fit(x, y)

print(f'params (a, b): {model.params}')
print(f'AIC-score: {model.get_aic()}')
