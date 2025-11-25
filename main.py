import numpy as np
import pandas as pd
from analisys import LinearModel, QuadraticModel, ExponentialModel 

x = [1, 2, 3, 4, 5]
y = [7.1, 8.9, 11.2, 12.8, 15.1]

models = [LinearModel(), QuadraticModel(), ExponentialModel()]
results = []

for model in models:
    model.fit(x, y)
    aic = model.get_aic()

    results.append({
        'Model': model.__class__.__name__,
        'AIC': round(aic, 2),        
        'Params': np.round(model.params, 3)
    })

df_results = pd.DataFrame(results).sort_values(by='AIC')
print(df_results)

