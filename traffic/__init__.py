# %%
import enum
import numpy as np
import pandas as pd

if __name__ == '__main__':
    import config
    timeit = lambda func: func
else:
    from . import config
    from utils import timeit


class TrafficType(enum.IntEnum):
    UNKNOWN = 0
    A = 1
    B = 2
    C = 3
    # D = 4


class TrafficModel:
    file_size = config.fileSize  # in bits
    delay_budgets = config.delayBudgets
    num_apps = config.numApps
    app_names = config.appNames
    period = 60 * 60 * 24 * 7  # a week (in seconds)
    
    profiles_path = config.profilesPath
    sample_rate = config.dpiSampleRate

    def __init__(self, area, period=period, scenario=None):
        self.area = area[0] * area[1] / 1e6  # km^2
        self.period = period
        self.scenario = scenario

    @classmethod
    def from_scenario(cls, scenario, area):
        if type(scenario) is str:
            scenario = TrafficType[scenario.upper()]
        else:
            scenario = TrafficType(scenario)
        profiles_df = pd.read_csv(cls.profiles_path, index_col=[0, 1, 2])
        profiles_df /=  cls.sample_rate
        profile = profiles_df.loc[int(scenario)]
        return cls(area, scenario=scenario).fit(profile)

    def fit(self, data_rate_df):
        """ Fit the traffic model to the given traffic trace dataset. """
        assert data_rate_df.shape[1] == self.num_apps
        df = data_rate_df[self.app_names]
        self.interval, rem = divmod(self.period, len(df))
        assert rem == 0, (self.interval, rem)
        self.rates = df * self.area / self.file_size  # files / s
        return self
    
    @property
    def time_slots(self):
        return self.rates.index

    @timeit
    def emit_traffic(self, time, dt):
        """ Randomly generate traffic for the given time.
        
        Returns:
        A list of length n_apps, each element is either (traffic-demand, delay-budget) or False.
        """
        return [(self.file_size, delay) if np.random.rand() < p else (None, None)
                for p, delay in zip(self.get_arrival_rates(time, dt), self.delay_budgets)]

    def _get_time_loc(self, time):
        return int((time / self.period) % 1 * len(self.rates))
    
    def get_time_slot(self, time):
        i = self._get_time_loc(time)
        return self.time_slots[i]
    
    def get_start_time_of_slot(self, time_slot):
        return self.interval * self.time_slots.get_loc(time_slot)
    
    def get_arrival_rates(self, time, dt):
        i = self._get_time_loc(time)
        return self.rates.values[i] * dt  # (n_apps)


# %%
if __name__ == '__main__':
    import plotly.express as px
    from plotly.subplots import make_subplots
    from collections import defaultdict
    
    area = (800, 800)
    area_km2 = area[0] * area[1] / 1e6
    figs = defaultdict(lambda: make_subplots(rows=len(TrafficType), cols=1, shared_xaxes=True))
    for i, traffic_type in enumerate(TrafficType):
        print(traffic_type)
        model = TrafficModel.from_scenario(traffic_type, area)
        thrp_df = model.rates * model.file_size / (1 << 20) / area_km2 # (Mb/(s*km^2))
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

# %%
