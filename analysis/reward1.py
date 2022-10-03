# %%
import numpy as np
import pandas as pd
import plotly.express as px

rho = np.linspace(0, 3, 100)
phi = np.linspace(1, 50, 100)
a = 1
b = 0.5
df = pd.DataFrame(np.array(np.meshgrid(rho, phi)).T.reshape(-1, 2), columns=['rho', 'phi'])
rho, phi = df['rho'], df['phi']
u = np.ones_like(rho)
u[rho >= 1] = b
# df['xi'] = 1 - phi ** (1 - rho)
# df['xi'] = np.tanh(1 - 1/rho) * (1 - 0.2*np.exp(-rho))
# df['xi'] = 2/(1+phi**(1/rho - 1)) - 1
df['xi'] = (rho - 1) * (rho < 1) * 1 + (1 - 1/(rho+1e-3)) * (rho >= 1) * 0.04

fig = px.line(df, x='rho', y='xi',
        # animation_frame='phi', animation_group='rho',
        range_y=[-1, 0.1])
fig.update_layout(
    xaxis_title=r'$\rho$',
    yaxis_title=r'$\xi$'
).show()
# px.line(df, x='rho', y='dxi_drho',
#         animation_frame='phi', animation_group='rho',
#         range_y=[-1, 1]).show()

# %%
