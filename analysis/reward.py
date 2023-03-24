# %%
import numpy as np
import pandas as pd
import plotly.express as px

rho = np.linspace(0, 3, 100)
phi = np.logspace(-3, 0, 15)
df = pd.DataFrame(np.array(np.meshgrid(rho, phi)).T.reshape(-1, 2), columns=['rho', 'phi'])
rho, phi = df['rho'], df['phi']
# df['xi'] = 1 - phi ** (1 - rho)
# df['xi'] = np.tanh(1 - 1/rho) * (1 - 0.2*np.exp(-rho))
# df['xi'] = 2/(1+phi**(1/rho - 1)) - 1
df['xi'] = (rho - 1) * (rho < 1) + (1 - 1/(rho+1e-3)) * (rho >= 1) * phi
df['$\\phi$'] = ['$10^{%.1f}$' % p for p in np.log10(phi)]
df = df.iloc[::-1]

fig = px.line(df, x='rho', y='xi',
              line_group='phi', color='$\\phi$',
              color_discrete_sequence=px.colors.qualitative.Pastel,
        # animation_frame='phi', animation_group='rho',
        range_y=[-1, 0.7])
fig.update_layout(
    xaxis_title=r'$\rho$',
    yaxis_title=r'$\xi$',
    font_size=13,
).show()
fig.write_image('reward.pdf', scale=2)
# px.line(df, x='rho', y='dxi_drho',
#         animation_frame='phi', animation_group='rho',
#         range_y=[-1, 1]).show()

# %%
