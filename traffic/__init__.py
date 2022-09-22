# %%
if __name__ != '__main__':
    from .model import TrafficType, TrafficModel
else:
    from model import *
    import plotly.express as px
    from plotly.subplots import make_subplots
    from collections import defaultdict

    figs = defaultdict(lambda: make_subplots(rows=len(TrafficType), cols=1, shared_xaxes=True))
    for i, traffic_type in enumerate(TrafficType):
        print(traffic_type)
        model = TrafficModel.from_scenario(traffic_type, sample_rate=1/300)
        model.print_info()
        for cat, rates in model.densities.items():
            days_idx = rates.index.get_level_values(0).unique()
            df = rates.unstack().reindex(days_idx)
            fig = px.imshow(df, title=traffic_type.name, labels=dict(x='time of day', y='day of week', color='Mb/s/km²'))
            print(cat)
            fig.show()
            figs[cat].add_trace(fig.data[0], row=i+1, col=1)
        print()
    for cat, fig in figs.items():
        fig.update_layout(height=600, width=600, title_text=cat)
        # fig.show()

# %%
