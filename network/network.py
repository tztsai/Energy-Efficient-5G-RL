# from xarray import DataArray
from utils import *
from . import config
from .env_utils import *
from .user_equipment import UserEquipment
from .base_station import BaseStation
from .channel import compute_channel_gain
from traffic import TrafficModel
from traffic.config import numApps, delayBudgets
from visualize.obs import anim_rolling
from config import *


class MultiCellNetwork:
    bss: Dict[int, BaseStation]
    ues: Dict[int, UserEquipment]
    
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

    buffer_ws = 16  # window size for computing recent arrival rates
    stats_save_path = 'analysis/'

    def __init__(self, area, bs_poses, traffic_scenario,
                 start_time=0, accelerate=1, dpi_sample_rate=None):
        self.area = area
        self.traffic_model = TrafficModel.from_scenario(
            traffic_scenario, area=area, sample_rate=dpi_sample_rate)
        self.start_time = self._parse_start_time(start_time)
        self.accelerate = accelerate
        self.bss = {}
        self.ues = {}
        self._bs_poses = None
        
        self.reset()
        
        for i, pos in enumerate(bs_poses):
            self.create_new_bs(i, pos)

        print('Initialized 5G multi-cell network: area={}, num_bs={}, scenario={}, start_time={}.'
              .format(self.area, self.num_bs, traffic_scenario, self.start_time))

    def reset(self):
        info('Resetting %s', self)
        if EVAL:
            vs = "energy arrived num done num_dropped dropped time service_time".split()
            self._eval_stats = pd.DataFrame(
                np.zeros((numApps, len(vs))), columns=vs)
            self._total_stats = self._eval_stats.copy()
            self._other_stats = defaultdict(list)
            self._stats_updated = True
        for bs in self.bss.values():
            bs.reset()
        self.ues.clear()
        self._time = 0
        self._timer = 0
        self._energy_consumed = 0
        self._buf_idx = 0
        self._arrival_buf = np.zeros((self.buffer_ws, numApps))
        self._stats = np.zeros((numApps, 3))

    def reset_stats(self):
        for bs in self.bss.values():
            bs.reset_stats()
        self._timer = 0
        self._energy_consumed = 0
        self._arrival_buf[self._buf_idx] = 0
        self._stats[:] = 0
        if EVAL:
            self._eval_stats[:] = 0
            self._stats_updated = False

    # @anim_rolling
    def update_stats(self):
        for bs in self.bss.values():
            bs.update_stats()
        if EVAL:
            self._eval_stats['arrived'] += self._arrival_buf[self._buf_idx]
            self._eval_stats['energy'] += self._energy_consumed
            self._total_stats += self._eval_stats
            self._stats_updated = True
        self._buf_idx = (self._buf_idx + 1) % self.buffer_ws

    @timeit
    def step(self, dt):
        self.generate_new_ues(dt)
    
        self.scan_connections()

        for bs in self.bss.values():
            bs.step(dt)

        for ue in list(self.ues.values()):
            ue.step(dt)
        
        self.update_timer(dt)
        
    def update_timer(self, dt):
        self._timer += dt
        self._time += dt

    def _parse_start_time(self, start_time):
        if type(start_time) is str:
            if start_time.isdigit():
                start_time = int(start_time)
            else:
                start_time = self.traffic_model.get_start_time_of_slot(start_time)
        assert isinstance(start_time, (int, float))
        return start_time  # in seconds
    
    @property
    def world_time(self):
        return self.start_time + self._time * self.accelerate
    
    @property
    def world_time_repr(self):
        m, s = divmod(round(self.world_time), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f'{calendar.day_abbr[d%7]}, {h:02}:{m:02}:{s:02}'

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
        return self._timer and self._energy_consumed / self._timer
    
    # @property
    # def avg_power_consumption(self):
    #     return self._time and self._total_energy_consumed / self._time
    
    @property
    @cache_obs
    def _recent_arrivals(self):
        return self._arrival_buf.mean(axis=0)
    
    @property
    def arrival_rates(self):
        if self._time:
            if DEBUG: assert self._stats_updated  # should only be called when _timer = step_time
            return self._recent_arrivals / self._timer
        return np.zeros(numApps)  # only before the first step
    
    # @property
    # def avg_arrival_rates(self):
    #     if self._time:
    #         return self._total_demand_buf / self._time
    #     return np.zeros(numApps)
    
    @property
    # @pd2np
    def drop_ratios(self):
        """ Drop ratios for each app category in the network. """
        return div0(self._stats[:, 1], self._recent_arrivals)

    @property
    # @pd2np
    def service_delays(self):
        return div0(self._stats[:, 2], self._stats[:, 0])

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

    def add_user(self, ue: UserEquipment):
        if DEBUG:
            assert ue.id not in self.ues, "UE %s already in the network" % ue.id
            # debug(f'{ue.id} added to the network')
        ue.net = self
        self.ues[ue.id] = ue
        self.measure_distances_and_gains(ue)
        self._arrival_buf[self._buf_idx, ue.app_type] += ue.demand

    def remove_user(self, ue_id):
        ue = self.ues.pop(ue_id)
        dropped = max(0., ue.demand)
        self._stats[ue.app_type] += [1, dropped, ue.delay]
        if DEBUG:
            if dropped:
                info('UE %s dropped' % ue_id)
        if EVAL:
            s, a = self._eval_stats, ue.app_type
            if dropped:
                s.at[a, 'dropped'] += dropped
                s.at[a, 'num_dropped'] += 1
            s.at[a, 'num'] += 1
            s.at[a, 'done'] += ue.file_size - dropped
            s.at[a, 'time'] += ue.delay
            s.at[a, 'service_time'] += ue.t_served
        #     record = [1, drop_ratio, ue.delay, ue.t_served]
        # else:
        #     record = [1, drop_ratio, ue.delay]
        # self._served_buf.iloc[ue.app_type] += record
        # self._buffer2[self._buf_idx, ue.app_type] += record
        # if ue.dropped:
        #     self._dropped[ue.app_type] += ue.demand / 1e6
        # self._delays[ue.app_type] += [ue.delay, 1]
        # if EVAL:
        #     if ue.done:
        #         self._total_done[ue.app_type] += [1, ue.served/1e6, ue.served_time, ue.delay]
        #     else:
        #         if ue.demand <= 0: breakpoint()
        #         self._total_dropped[ue.app_type] += [1, ue.demand/1e6, ue.served_time, ue.delay]

    @timeit
    def scan_connections(self):
        for ue in self.ues.values():
            if ue.idle:
                ue.request_connection()

    @timeit
    def measure_distances_and_gains(self, ue):
        ue.distances = np.sqrt(np.sum((self.bs_positions - ue.pos)**2, axis=1))
        ue.channel_gains = compute_channel_gain(ue.distances)

    @timeit
    def generate_new_ues(self, dt, **kwargs):
        new_traffic = self.traffic_model.emit_traffic(self.world_time, dt)
        for app, (demand, delay) in enumerate(new_traffic):
            if not demand: continue
            if 'pos' not in kwargs:
                kwargs['pos'] = np.append(np.random.rand(2) * self.area, UserEquipment.height)
            ue = UserEquipment(app_type=app, demand=demand, delay_budget=delay, **kwargs)
            self.add_user(ue)

    def consume_energy(self, energy):
        self._energy_consumed += energy

    def set_action(self, bs_id, action):
        self.bss[bs_id].take_action(action)

    def observe_bs(self, bs_id):
        return self.bss[bs_id].get_observation()

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
            self.drop_ratios,            # dropped rates in different delay cats (3)
            self.service_delays,        # avg delay in different delay cats (3)
            self.arrival_rates,         # rates demanded by new UEs in different delay cats (3)
            [thrp, thrp_req, log_ratio],# throughput (3)
            bs_obs                      # bs observations
        ], dtype=np.float32)

    __info_dict_src = re.sub(r'\n\s*', '\n', """
        # current step info
        time = self.world_time_repr
        pc = self.power_consumption  # W
        drop_ratios = self.drop_ratios
        delays = self.service_delays * 1e3  # ms
        actual_rate = sum(ue.data_rate for ue in self.ues.values()) / 1e6  # Mb/s
        required_rate = sum(ue.required_rate for ue in self.ues.values()) / 1e6
        arrival_rates = div0(self._eval_stats.arrived.values, self._timer) / 1e6
        arrival_rate = np.sum(arrival_rates)
        # all steps stats
        total_time = self._time
        total_counts = self._total_stats.num.values
        total_dropped_counts = self._total_stats.num_dropped.values
        total_arrived = self._total_stats.arrived.values / 1e6  # Mb
        total_done = self._total_stats.done.values / 1e6
        total_dropped = self._total_stats.dropped.values / 1e6
        total_quitted = total_done + total_dropped
        total_energy = self._total_stats.energy.values[0]  # J
        avg_pc = div0(total_energy, total_time)  # W
        avg_drop_sizes = div0(total_dropped, total_dropped_counts)
        avg_drop_ratios = div0(total_dropped, total_quitted)
        avg_delays = div0(self._total_stats.time.values * 1e3, total_counts)
        avg_arrival_rates = div0(total_arrived, total_time) 
        avg_demand_sizes = div0(total_quitted, total_counts)
        avg_sum_rates = div0(total_done, total_time)
        avg_ue_rates = div0(total_done, self._total_stats.time.values)
        avg_energy_efficiency = div0(total_done, total_energy)  # Mb/J
        avg_service_time_ratios = div0(self._total_stats.service_time.values,
                                       self._total_stats.time.values)
    """)

    @cache_obs
    def info_dict(self, include_bs=True, _s=__info_dict_src):
        # assert self._stats_updated
        
        infos = {}
        scope = locals()
        scope.update(globals())
        exec(_s, scope, infos)
        
        if include_bs:
            for i, bs in self.bss.items():
                for k, v in bs.info_dict().items():
                    infos[f'bs_{i}_{k}'] = v

        return infos
    
    @classmethod
    def annotate_obs(cls, obs):
        keys = ['power_consumption',
                *[f'drop_rate_cat{i}' for i in range(3)],
                *[f'delay_cat{i}' for i in range(3)],
                *[f'arrival_rate_cat{i}' for i in range(3)],
                'sum_rate', 'sum_rate_req', 'rate_log_ratio',
                *[f'bs{i}_obs{j}' for i in range(config.numBS) for j in range(cls.bs_obs_ndims)]]
        assert len(keys) == len(obs)
        return dict(zip(keys, obs))

    def add_stat(self, key, val):
        self._other_stats[key].append(val)
    
    def save_other_stats(self):
        for k, v in self._other_stats.items():
            pd.DataFrame(v).to_csv(f'{self.stats_save_path}/{k}.csv', index=False)

    def __repr__(self) -> str:
        return 'Network'
        return '{}({})'.format(
            type(self).__name__,
            kwds_str(area=self.area, num_bs=self.num_bs,
                     scenario=self.traffic_model.scenario.name,
                     start_time=' '.join(self.traffic_model.get_time_slot(self.start_time)))
        )
