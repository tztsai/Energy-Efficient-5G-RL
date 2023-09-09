# %%
import pandas as pd
from pathlib import Path

cells_df = pd.read_excel('cells.xlsx')
# cells_df = cells_df[~cells_df.range.isnull()]
cells_df

# %%
data_folder = Path('cells-traffic')
sql_url = 'sqlite:///cell_traffic.sql'

# %%
from tqdm.notebook import tqdm

for i in tqdm(cells_df.cell_id):
    path = data_folder / '{}.csv'.format(i)
    if not path.exists(): continue
    df = pd.read_csv(path, index_col=0)
    if len(df) < 50: continue
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.to_sql(name=str(i), con=sql_url, if_exists='replace')

# %%
