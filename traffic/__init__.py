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
    MIXED = 1
    # *** USE THE TRAFFIC TYPES BELOW FOR TRAINING ***
    RESIDENT = 2
    URBAN = 3
    OFFICE = 4


class TrafficModel:
    file_size = config.fileSize  # in bits
    delay_budgets = config.delayBudgets
    num_apps = config.numApps
    period = 60 * 60 * 24 * 7  # a week (in seconds)
    
    profiles_path = config.profilesPath
    sample_rate = config.dpiSampleRate

    def __init__(self, period=period, scenario=None):
        self.period = period
        self.scenario = scenario

    @classmethod
    def from_scenario(cls, scenario):
        if type(scenario) is str:
            scenario = TrafficType[scenario.upper()]
        else:
            scenario = TrafficType(scenario)
        profiles_df = pd.read_csv(cls.profiles_path, index_col=[0, 1, 2])
        profiles_df /= cls.sample_rate
        profile = profiles_df.loc[int(scenario)]
        return cls(scenario=scenario).fit(profile)

    def fit(self, data_rate_df):
        """ Fit the traffic model to the given traffic trace dataset. """
        assert data_rate_df.shape[1] == self.num_apps
        self.interval, rem = divmod(self.period, len(data_rate_df))
        assert rem == 0, (self.interval, rem)
        self.rates = data_rate_df / self.file_size  # files per second
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
    for traffic_type in TrafficType:
        print(traffic_type)
        model = TrafficModel.from_scenario(traffic_type)
        thrp_df = model.rates * model.file_size / (1 << 20) # throughput (MBps)
        print(thrp_df.describe())
        peak_time = model.rates.sum(axis=1).idxmax()
        peak_time_secs = model.get_start_time_of_slot(peak_time)
        print(peak_time, peak_time_secs)
        # for cat, rates in thrp_df.items():
        #     days_idx = rates.index.get_level_values(0).unique()
        #     df = rates.unstack().reindex(days_idx)
        #     px.imshow(df, title=f'{traffic_type.name} {cat}').show()
        print()
