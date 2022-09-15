# %%
if __name__ != '__main__':
    from .model import TrafficType, TrafficModel
else:
    from model import *
    import plotly.express as px
    from plotly.subplots import make_subplots
    from collections import defaultdict

    area = (1000, 1000)
    area_km2 = area[0] * area[1] / 1e6
    figs = defaultdict(lambda: make_subplots(rows=len(TrafficType), cols=1, shared_xaxes=True))
    for i, traffic_type in enumerate(TrafficType):
        print(traffic_type)
        model = TrafficModel.from_scenario(traffic_type, area)
        thrp_df = model.rates * model.file_size / (1 << 20) / area_km2 # (Mb/(s*km^2))
        print(model.rates.max())
        thrp_df['Total'] = thrp_df.sum(axis=1)
        print(thrp_df.describe())
        peak_time = model.rates.sum(axis=1).idxmax()
        peak_time_secs = model.get_start_time_of_slot(peak_time)
        print(peak_time, peak_time_secs)
        for cat, rates in thrp_df.items():
            days_idx = rates.index.get_level_values(0).unique()
            df = rates.unstack().reindex(days_idx)
            fig = px.imshow(df, title=traffic_type.name, labels=dict(x='time of day', y='day of week', color='Mb/(s⋅km²)'))
            print(cat)
            # fig.show()
            figs[cat].add_trace(fig.data[0], row=i+1, col=1)
        print()
    for cat, fig in figs.items():
        fig.update_layout(height=600, width=600, title_text=cat)
        # fig.show()
