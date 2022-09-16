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
    sample_rate = config.dpiSampleRate
    profiles_path = config.profilesPath

    def __init__(self, area=None, period=period, scenario=None, sample_rate=None):
        self.area = 1 if area is None else area[0] * area[1] / 1e6  # km^2
        self.period = period
        self.scenario = scenario
        if sample_rate is not None:
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
        self.density_df = density_df = df / 1e6
        density_df['Total'] = density_df.sum(axis=1)
        info_df = density_df.describe()
        info_df.loc['highest time'] = density_df.idxmax(axis=0)
        info_df.loc['lowest time'] = density_df.idxmin(axis=0)
        notice('Traffic scenario: {}'.format(self.scenario.name))
        notice(str(info_df) + '\n')
        assert rem == 0, (self.interval, rem)
        self.rates = df * self.area / self.file_size  # files / s
        assert self.rates.max().max() * 1e-3 <= 1.
        notice(str(self.rates.describe()) + '\n')
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
