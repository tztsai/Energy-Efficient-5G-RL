import re
import enum
import numpy as np
import pandas as pd

try:
    from . import config
    from utils import timeit, notice
except ImportError:
    import config
    timeit = lambda func: func
    notice = print


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
    sample_rates = config.dpiSampleRates
    profiles_path = config.profilesPath

    def __init__(self, area=None, period=period, scenario=None, sample_rate=None):
        self.area = 1 if area is None else area[0] * area[1] / 1e6  # km^2
        self.period = period
        self.scenario = scenario
        if sample_rate is None:
            if scenario is None:
                self.sample_rate = 1
            else:
                self.sample_rate = self.sample_rates[scenario.value - 1]
        else:
            self.sample_rate = sample_rate

    @classmethod
    def from_scenario(cls, scenario, **kwargs):
        if type(scenario) is str:
            scenario = TrafficType[scenario.upper()]
        else:
            scenario = TrafficType(scenario)
        profiles_df = pd.read_csv(cls.profiles_path, index_col=[0, 1, 2])
        profile = profiles_df.loc[int(scenario)]
        return cls(scenario=scenario, **kwargs).fit(profile)

    def fit(self, data_rate_df):
        """ Fit the traffic model to the given traffic trace dataset. """
        assert data_rate_df.shape[1] == self.num_apps
        df = data_rate_df[self.app_names] / self.sample_rate
        self.interval, rem = divmod(self.period, len(df))
        assert rem == 0, (self.interval, rem)
        self.rates = df * self.area / self.file_size  # files / s
        assert self.rates.values.max() * 1e-3 <= 1.
        
        df = pd.concat([df / 1e6, self.rates], axis=1, keys=['Mb/s/km^2', 'files/s'])
        total_df = df.groupby(level=0, axis=1).sum()
        for k in df.columns.levels[0]:
            df[k, 'Total'] = total_df[k]
        info_df = df.describe().T[['mean', 'std', 'min', 'max']].sort_index()
        info_df['peak time'] = df.idxmax(axis=0).map(lambda x: f'{x[0]}, {x[1]}')
        info_df['vale time'] = df.idxmin(axis=0).map(lambda x: f'{x[0]}, {x[1]}')
        self.info_df = info_df.set_index(pd.MultiIndex.from_tuples(info_df.index))
        
        return self
    
    def print_info(self):
        notice('Traffic scenario: %s', self.scenario.name)
        notice('%s\n', self.info_df)
        
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
    
    def get_start_time_of_slot(self, time_slot: str):
        m = re.match(r'(\w{3,}),?\s*(\d+):(\d+)', time_slot)
        d, h, m = m[1][:3], int(m[2]), int(m[3])
        s = d, f"{h:02d}:{m:02d}"
        return self.interval * self.time_slots.get_loc(s)
    
    def get_arrival_rates(self, time, dt):
        i = self._get_time_loc(time)
        return self.rates.values[i] * dt  # (n_apps)
