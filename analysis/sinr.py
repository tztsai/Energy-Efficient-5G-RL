# %%
import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('sinr.csv')
df

# %%
df['R'] = 20e6 * np.log2(1 + df.SINR)
df['p_rc'] = df.p * df.g

for k in ['g', 'S', 'I', 'SINR']:
    df[k] = 10 * np.log10(df[k] + 1e-6)

# %%
px.scatter_3d(df, x='I', y='S', z='R', color='SINR')

# %%
nx, ny = 81, 81
gridx, gridy = np.mgrid[0:1000:nx*1j, 0:1000:nx*1j]
grid = np.c_[gridx.flat, gridy.flat]
grid

# %%
from scipy.spatial import cKDTree
from sklearn.impute import KNNImputer as Imputer

tree = cKDTree(grid)
dd, ii = tree.query(df[['x', 'y']])
df1 = df.set_index(ii).groupby(level=0).mean()
df1

# %%
new_df = pd.DataFrame(grid, columns=['x', 'y'])
features = ['g', 'S', 'I', 'SINR']
new_df[features] = df1[features]
new_df[:] = Imputer().fit_transform(new_df)

# %%
titles = dict(g='Channel Gain', S='Signal Power', I='Interference', SINR='SINR')

for f in features:
    im = new_df.pivot('y', 'x', f)
    z = im.values.reshape(-1)
    fig = px.imshow(im, range_color=np.percentile(z, [2, 96]), title=titles[f], labels=dict(x='x (m)', y='y (m)', color='dB'))
    # fig.write_image(f'../results/plots/grid_{f}.png', scale=2)
    fig.show()
    # fig = px.density_contour(new_df, x='x', y='y', z=f)
# im = sinr_df.sinr.values.reshape(nx, ny)

# px.imshow(im, zmin=-25, origin='lower')

# %%
