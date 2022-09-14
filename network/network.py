from utils import *
from . import config
from .env_utils import *
from .user_equipment import User
from .base_station import BaseStation
from .channel import compute_channel_gain
from traffic import TrafficModel
from config import *


class MultiCellNetwork:
    inter_bs_dist = config.interBSDist
    default_area = config.areaSize
    default_bs_poses = config.bsPositions

    global_obs_space = make_box_env([[0, np.inf]] * (1 + 3 + 3 + 3 + 3))
    bs_obs_space = BaseStation.total_obs_space
    net_obs_space = concat_box_envs(
        global_obs_space,
        duplicate_box_env(BaseStation.self_obs_space, config.numBS),
        duplicate_box_env(BaseStation.mutual_obs_space,
                          config.numBS * (config.numBS - 1) // 2))

    global_obs_ndims = box_env_ndims(global_obs_space)
    bs_obs_ndims = box_env_ndims(bs_obs_space)
    net_obs_ndims = box_env_ndims(net_obs_space)
    
    def __init__(self, area, bs_poses, traffic_type, start_time=0, accel_rate=1):
        self.area = area
        self.traffic_model = TrafficModel.from_scenario(traffic_type, area)
        self.start_time = self._parse_start_time(start_time)
        self.accel_rate = accel_rate
        self.bss = {}
        self.ues = {}
        self._bs_poses = None
        self.reset()
        for i, pos in enumerate(bs_poses):
            self.create_new_bs(i, pos)
        print('Initialized green 5G network: area={}, num_bs={}, scenario={}, start_time={}.'
              .format(self.area, self.num_bs, traffic_type, self.start_time))

    def _parse_start_time(self, start_time):
        if type(start_time) is str:
            if start_time.isdigit():
                start_time = int(start_time)
            else:
                start_time = self.traffic_model.get_start_time_of_slot(start_time)
        return start_time  # in seconds
    
    @property
    def world_time(self):
        return self.start_time + self._time * self.accel_rate
    
    @property
    def world_time_tuple(self):
        m, s = divmod(self.world_time, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return int(d % 7), int(h), int(m), s

    @property
    def time_slot(self):
        return self.traffic_model.get_time_slot(self.world_time)
    
    # @property
    # def ue_counts(self):
    #     return np.bincount(np.array([ue.app_type for ue in self.ues.values()]),
    #                        minlength=self.traffic_model.num_apps)

    @property
    def num_bs(self):
        return len(self.bss)
    
    @property
    def num_ue(self):
        return len(self.ues)
    
    @property
    def bs_positions(self):
        if self._bs_poses is None:
            self._bs_poses = np.array([self.bss[i].pos for i in range(self.num_bs)])
        return self._bs_poses
    
    @property
    def power_consumption(self):
        """ Power consumption of all BSs in the network in kW. """
        return self._timer and self._energy_consumed / self._timer / 1e3
    
    @property
    def avg_power_consumption(self):
        return self._time and self._total_energy_consumed / self._time / 1e3
    
    @property
    def arrival_rates(self):
        if self._timer:
            return self._demand / self._timer
        return np.zeros_like(self._demand)
    
    @property
    def drop_rates(self):
        """ Drop rates for each app category in the network in mb/s. """
        if self._timer:
            return self._dropped / self._timer
        return np.zeros_like(self._dropped)
    
    @property
    def service_delays(self):
        return self._delays[:,0] / np.maximum(self._delays[:,1], 1)

    def get_bs(self, id):
        return self.bss[id]

    def get_ue(self, id):
        return self.ues[id]

    def add_base_station(self, bs):
        assert bs.id not in self.bss, "BS %d already in the network" % bs.id
        bs.net = self
        self.bss[bs.id] = bs
        info(f'{bs} added to the network')

    def create_new_bs(self, id, pos, **kwargs):
        pos = np.append(pos, BaseStation.bs_height)
        bs = BaseStation(id, pos, net=self, **kwargs)
        self.add_base_station(bs)
        return bs

    def add_user(self, ue):
        if DEBUG:
            assert ue.id not in self.ues, "UE %s already in the network" % ue.id
            # debug(f'UE({ue.id}, cat={ue.app_type}) added to the network')
        ue.net = self
        self.ues[ue.id] = ue
        self.measure_distances_and_gains(ue)
        self._demand[ue.app_type] += ue.demand / 1e6

    @timeit
    def scan_connections(self):
        for ue in self.ues.values():
            if ue.idle:
                ue.request_connection()

    @timeit
    def measure_distances_and_gains(self, ue):
        ue.distances = np.sqrt(np.sum((self.bs_positions - ue.pos)**2, axis=1))
        # debug(f'Distances of UE {ue.id} to BSs: {ue.dists}')
        ue.channel_gains = compute_channel_gain(ue.distances)

    def remove_user(self, ue_id):
        ue = self.ues[ue_id]
        del self.ues[ue_id]
        if ue.dropped:
            self._dropped[ue.app_type] += ue.demand / 1e6
        self._delays[ue.app_type] += [ue.delay, 1]
        if EVAL:
            if ue.done:
                self._total_done[ue.app_type] += [1, ue.served/1e6, ue.delay]
            else:
                if ue.demand <= 0: breakpoint()
                self._total_dropped[ue.app_type] += [1, ue.demand/1e6, ue.delay]

    @timeit
    def generate_new_ues(self, dt, **kwargs):
        new_traffic = self.traffic_model.emit_traffic(self.world_time, dt)
        for app, (demand, delay) in enumerate(new_traffic):
            if not demand: continue
            if 'pos' not in kwargs:
                xy = np.random.rand(2) * self.area
                kwargs['pos'] = np.append(xy, User.ue_height)
            ue = User(app_type=app, demand=demand, delay_budget=delay, **kwargs)
            self.add_user(ue)

    def consume_energy(self, energy):
        self._energy_consumed += energy

    # def add_stat(self, key, val, dt):
    #     self._other_stats[key] += [1, val, dt]

    def reset(self):
        info('Resetting %s', self)
        for bs in self.bss.values():
            bs.reset()
        self.ues.clear()
        self._time = 0
        self._demand = np.zeros(self.traffic_model.num_apps)
        self._dropped = np.zeros(self.traffic_model.num_apps)
        self._delays = np.zeros((self.traffic_model.num_apps, 2))
        self._total_dropped = np.zeros((self.traffic_model.num_apps, 3))
        self._total_done = np.zeros((self.traffic_model.num_apps, 3))
        self._total_energy_consumed = 0
        self._other_stats = defaultdict(lambda: np.zeros(3, dtype=np.float32))
        self.reset_stats()

    @timeit
    def step(self, dt):
        self.generate_new_ues(dt)
    
        self.scan_connections()

        for bs in self.bss.values():
            bs.step(dt)

        for ue in list(self.ues.values()):
            ue.step(dt)
        
        self.update_timer(dt)

    def set_action(self, bs_id, action):
        self.bss[bs_id].take_action(action)

    def observe_bs(self, bs_id):
        return self.bss[bs_id].get_observation()

    def update_timer(self, dt):
        self._timer += dt
        self._time += dt

    def reset_stats(self):
        for bs in self.bss.values():
            bs.reset_stats()
        self._demand[:] = 0
        self._dropped[:] = 0
        self._delays[:] = 0
        self._energy_consumed = 0
        self._timer = 0

    def update_stats(self):
        for bs in self.bss.values():
            bs.update_stats()
        if EVAL:
            self._total_energy_consumed += self._energy_consumed
    
    @cache_obs
    def observe_network(self):
        bs_obs = []
        for i in range(self.num_bs):
            bs_obs.append(self.bss[i].observe_self())
        for i in range(self.num_bs):
            for j in range(i):
                bs_obs.append(self.bss[i].observe_other(self.bss[j])[0])
        bs_obs = np.concatenate(bs_obs, dtype=np.float32)
        thrp, thrp_req, log_ratio = BaseStation.calc_sum_rate(self.ues.values())
        return np.concatenate([
            [self.power_consumption],   # power consumption (1)
            self.drop_rates,            # dropped rates in different delay cats (3)
            self.service_delays,        # avg delay in different delay cats (3)
            self.arrival_rates,         # rates demanded by new UEs in different delay cats (3)
            [thrp, thrp_req, log_ratio],# throughput (3)
            bs_obs                      # bs observations
        ], dtype=np.float32)

    @cache_obs
    def info_dict(self):
        obs = self.observe_network()
        d, h, m, s = self.world_time_tuple
        return dict(
            time='{}, {:02}:{:02}:{:02}'.format(calendar.day_abbr[d], h, m, int(s)),
            power_consumption=obs[0],
            dropped_rate=obs[1:4].sum(),
            delay=obs[4:7].mean(),
            arrival_rate=obs[7:10].sum(),
            actual_rate=obs[10],
            required_rate=obs[11],
            avg_pc=self.avg_power_consumption,
            total_done_vol=self._total_done[:, 1],
            total_dropped_vol=self._total_dropped[:, 1],
            avg_serve_time=1000 * (self._total_done[:, 2] + self._total_dropped[:, 2]) /
                (self._total_done[:, 0] + self._total_dropped[:, 0] + 1e-6),
            avg_data_rates=self._total_done[:, 1] /
                np.maximum(self._total_done[:, 2], 1e-6),
            avg_drop_rates=self._total_dropped[:, 1] /
                np.maximum(self._total_dropped[:, 2], 1e-6),
            total_done_count=self._total_done[:, 0].sum(),
            total_dropped_count=self._total_dropped[:, 0].sum(),
            total_drop_ratios=self._total_dropped[:, 1] / 
                (self._total_done[:, 1] + self._total_dropped[:, 1] + 1e-6),
            bs_info=pd.DataFrame({i: bs.info_dict() for i, bs in self.bss.items()}).T,
            # ue_info=pd.DataFrame({i: ue.info_dict() for i, ue in self.ues.items()}).T
        )
        
    def annotate_obs(self, obs):
        keys = ['power_consumption',
                *[f'arrival_rate_cat{i}' for i in range(3)],
                *[f'drop_rate_cat{i}' for i in range(3)],
                'sum_rate', 'sum_rate_req', 'rate_log_ratio',
                *[f'bs{i}_obs{j}' for i in range(self.num_bs) for j in range(self.bs_obs_ndims)]]
        assert len(keys) == len(obs)
        return dict(zip(keys, obs))
        
    def __repr__(self) -> str:
        if not DEBUG:
            return 'Network'
        return '{}({})'.format(
            type(self).__name__,
            kwds_str(area=self.area, num_bs=self.num_bs,
                     scenario=self.traffic_model.scenario.name,
                     start_time=self.traffic_model.get_time_slot(self.start_time))
        )
