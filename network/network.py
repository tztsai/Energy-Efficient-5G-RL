# from xarray import DataArray
from utils import *
from . import config
from .env_utils import *
from .user_equipment import UserEquipment, TestProbe
from .base_station import BaseStation
from .channel import compute_channel_gain
from traffic import TrafficModel, TrafficType
from traffic.config import numApps, delayBudgets
from visualize.obs import anim_rolling
from config import *


class MultiCellNetwork:
    bss: OrderedDict[int, BaseStation]
    ues: Dict[int, UserEquipment]
    
    inter_bs_dist = config.interBSDist
    default_area = config.areaSize
    default_bs_poses = config.bsPositions
    default_scenario = 'RANDOM'

    global_obs_space = make_box_env([[0, np.inf]] * (1 + 4 + 4))
    bs_obs_space = BaseStation.total_obs_space
    #  pruned_bs_space = concat_box_envs(
    #     BaseStation.self_obs_space,
    #     duplicate_box_env(BaseStation.mutual_obs_space, config.numBS - 1))
    net_obs_space = concat_box_envs(
        global_obs_space,
        duplicate_box_env(bs_obs_space, config.numBS))

    global_obs_dim = box_env_ndims(global_obs_space)
    bs_obs_dim = box_env_ndims(bs_obs_space)
    net_obs_dim = box_env_ndims(net_obs_space)

    buffer_ws = 16  # window size for computing recent arrival rates
    stats_save_path = 'analysis/'

    def __init__(self,
                 traffic_scenario=default_scenario,
                 area=default_area,
                 bs_poses=default_bs_poses, 
                 start_time=0,
                 accelerate=1,
                 dpi_sample_rate=None):
        self.area = area
        self.traffic_scenario = traffic_scenario
        self.accelerate = accelerate
        self.bss = OrderedDict()
        self.ues = {}
        self._bs_poses = None
        self._csi_cache = {}
        self._make_traffic_model = partial(
            TrafficModel.from_scenario, area=area, sample_rate=dpi_sample_rate)

        if traffic_scenario in TrafficType._member_names_:
            self.traffic_model = self._make_traffic_model(traffic_scenario)
        else:
            self.traffic_model = None
        self.start_time = self._parse_start_time(start_time)
        
        self.reset()

        for i, pos in enumerate(bs_poses):
            self.create_new_bs(i, pos)

    def reset(self):
        if EVAL:
            vs = "energy arrived num done num_dropped dropped time service_time".split()
            self._arrival_buf = np.zeros((self.buffer_ws, numApps))
            self._eval_stats = pd.DataFrame(
                np.zeros((numApps, len(vs))), columns=vs)
            self._total_stats = defaultdict(float)
            self._other_stats = defaultdict(list)
            self._stats_updated = True
        if self.traffic_model is None or self.traffic_scenario == 'RANDOM':
            self.traffic_model = self._make_traffic_model(self.traffic_scenario)
        for bs in self.bss.values():
            bs.reset()
        self.ues.clear()
        self._time = 0
        self._timer = 0
        self._energy_consumed = 0
        self._buf_idx = 0
        # self._stats = np.zeros((numApps, 3))
        self.quitted_stats = np.zeros((2, 2))
        notice('Reset %s', repr(self))

    def reset_stats(self):
        for bs in self.bss.values():
            bs.reset_stats()
        self._timer = 0
        self._energy_consumed = 0
        self.quitted_stats[:] = 0
        if EVAL:
            self._arrival_buf[self._buf_idx] = 0
            self._eval_stats[:] = 0
            self._stats_updated = False

    # @anim_rolling
    def update_stats(self):
        for bs in self.bss.values():
            bs.update_stats()
        if EVAL:
            self._eval_stats['arrived'] += self._arrival_buf[self._buf_idx]
            self._eval_stats['energy'] += self._energy_consumed
            for k, v in self._eval_stats.items():
                self._total_stats[k] += v.values
            self._buf_idx = (self._buf_idx + 1) % self.buffer_ws
            self._stats_updated = True

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
        rep = f'{calendar.day_abbr[d%7]}, {h:02}:{m:02}'
        if s: rep += f':{s:02}'
        return rep

    @property
    def time_slot(self):
        return self.traffic_model.get_time_slot(self.world_time)
    
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
    
    @property
    def arrival_rates(self):
        if self._time:
            if DEBUG and EVAL:
                assert self._stats_updated  # should only be called when _timer = step_time
            return self._arrival_buf.mean(axis=0) / self._timer
        return np.zeros(numApps)  # only before the first step
    
    @property
    def drop_ratios(self):
        """ Ratios of dropped traffic for each app category in the current step. """
        return div0(self.quitted_stats[:, 1], self.quitted_stats[:, 0] * self.traffic_model.file_size)

    @property
    def service_delays(self):
        """ Average service delays per UE for each app category in the current step. """
        return div0(self.quitted_stats[:, -1], self.quitted_stats[:, 0])
    
    @property
    def avg_rate_ratios(self):
        """ r_avg/r_req for finished UEs and dropped UEs. """
        return (self.quitted_stats[:, 1] + 1e-6) / (self.quitted_stats[:, 0] + 1e-6)

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

    def add_user(self, ue: UserEquipment=None, **kwargs):
        if ue is None:
            ue = UserEquipment(**kwargs)
        if DEBUG:
            assert ue.id not in self.ues, "UE %s already in the network" % ue.id
            # debug(f'{ue.id} added to the network')
        ue.net = self
        self.ues[ue.id] = ue
        self.measure_distances_and_gains(ue)
        if EVAL:
            self._arrival_buf[self._buf_idx, ue.service] += ue.demand

    def remove_user(self, ue_id):
        ue = self.ues.pop(ue_id)
        dropped = ue.demand
        # self._stats[ue.service] += [1, dropped, ue.delay]
        is_dropped = int(dropped > 0)
        if is_dropped:
            s = dropped / ue.total_demand  # drop ratio
        else:
            s = ue.delay / ue.delay_budget  # delay ratio
        self.quitted_stats[is_dropped] += [1, s]
        if DEBUG:
            if dropped:
                assert dropped > 0.
                info('UE %s dropped' % ue_id)
        if EVAL:
            s, a = self._eval_stats, ue.service
            if dropped:
                s.at[a, 'dropped'] += dropped
                s.at[a, 'num_dropped'] += 1
            s.at[a, 'num'] += 1
            s.at[a, 'done'] += ue.total_demand - dropped
            s.at[a, 'time'] += ue.delay
            s.at[a, 'service_time'] += ue.t_served

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
        for service, (demand, delay) in enumerate(new_traffic):
            if not demand: continue
            if 'pos' not in kwargs:
                kwargs['pos'] = np.append(np.random.rand(2) * self.area, UserEquipment.height)
            self.add_user(service=service, demand=demand, delay_budget=delay, **kwargs)
            
    def test_network_channel(self):
        state = tuple((bs.num_ant, bs.num_ue, bs.responding, bs.sleep > 0)
                      for bs in self.bss.values())
        cache = self._csi_cache
        if state not in cache:
            cache[state] = TestProbe(self).test_sinr()
        return cache[state]

    def consume_energy(self, energy):
        self._energy_consumed += energy

    def set_action(self, bs_id, action):
        self.bss[bs_id].take_action(action)

    def observe_bs(self, bs_id):
        return self.bss[bs_id].get_observation()

    @cache_obs
    def observe_network(self):
        bs_obs = [self.observe_bs(i) for i in range(self.num_bs)]
        # for i in range(self.num_bs):
        #     bs_obs.append(self.bss[i].get_observation())
        # for i in range(self.num_bs):
        #     for j in range(i):
        #         bs_obs.append(self.bss[i].observe_other(self.bss[j])[0])
        bs_obs = np.concatenate(bs_obs, dtype=np.float32)
        thrp = 0.
        req_thrps = np.zeros(3)
        for ue in self.ues.values():
            req_thrps[ue.status] += ue.required_rate
            thrp += ue.data_rate
        return np.concatenate([
            [self.power_consumption],       # power consumption (1)
            self.quitted_stats.reshape(-1), # stats of quitted UEs last step (4)
            # self.drop_ratios,           # drop rates in different delay cats (3)
            # self.service_delays,        # avg delay in different delay cats (3)
            # self.arrival_rates,         # rates demanded by new UEs in different delay cats (3)
            [*req_thrps, thrp],         # required (idle, queued, active) and actual sum rates (4)
            bs_obs                      # bs observations
        ], dtype=np.float32)

    @cache_obs
    def info_dict(self, include_bs=False):
        # assert self._stats_updated
        
        infos = dict(
            time=self.world_time_repr,
            pc=self.power_consumption,  # W
            drop_ratios=self.drop_ratios,
            delays=self.service_delays * 1e3,  # ms
            actual_rate=sum(ue.data_rate for ue in self.ues.values()) / 1e6,  # Mb/s
            required_rate=sum(ue.required_rate for ue in self.ues.values()) / 1e6,
            arrival_rates=div0(self._eval_stats.arrived.values, self._timer) / 1e6,
            arrival_rate=div0(self._eval_stats.arrived.values.sum(), self._timer) / 1e6
        )
        
        if include_bs:
            for i, bs in self.bss.items():
                for k, v in bs.info_dict().items():
                    infos[f'bs_{i}_{k}'] = v

        return infos
    
    def calc_total_stats(self):
        for bs in self.bss.values():
            bs.calc_total_stats()

        _vars = set(list(vars()) + ['_vars'])
        
        total_time = self._time
        total_counts = self._total_stats['num']
        total_dropped_counts = self._total_stats['num_dropped']
        total_arrived = self._total_stats['arrived'] / 1e6  # Mb
        total_done = self._total_stats['done'] / 1e6
        total_dropped = self._total_stats['dropped'] / 1e6
        total_quitted = total_done + total_dropped
        total_energy = self._total_stats['energy'][0]  # J
        avg_pc = div0(total_energy, total_time)  # W
        avg_drop_sizes = div0(total_dropped, total_dropped_counts)
        avg_drop_ratios = div0(total_dropped, total_quitted)
        avg_delays = div0(self._total_stats['time'] * 1e3, total_counts)
        avg_arrival_rates = div0(total_arrived, total_time)
        avg_demand_sizes = div0(total_quitted, total_counts)
        avg_sum_rates = div0(total_done, total_time)
        avg_ue_rates = div0(total_done, self._total_stats['time'])
        avg_energy_efficiency = div0(total_done, total_energy)  # Mb/J
        avg_service_time_ratios = div0(self._total_stats['service_time'],
                                       self._total_stats['time'])
        
        self._total_stats.update(it for it in vars().items() if it[0] not in _vars)
    
    @classmethod
    def annotate_obs(cls, obs):
        keys = ['power_consumption',
                *[f'drop_rate_cat{i}' for i in range(3)],
                *[f'delay_cat{i}' for i in range(3)],
                *[f'arrival_rate_cat{i}' for i in range(3)],
                'sum_rate', 'sum_rate_req', 'rate_log_ratio',
                *[f'bs{i}_obs{j}' for i in range(config.numBS) 
                  for j in range(cls.bs_obs_dim)]]
        assert len(keys) == len(obs)
        return dict(zip(keys, obs))

    def add_stat(self, key, val):
        self._other_stats[key].append(val)

    def save_stats(self):
        self.calc_total_stats()
        pd.Series(self._total_stats).to_csv(
            f'{self.stats_save_path}/network_stats.csv', header=False)
        for k, v in self._other_stats.items():
            pd.DataFrame(v).to_csv(f'{self.stats_save_path}/{k}.csv', index=False)

    def __repr__(self) -> str:
        return '{}({})'.format(
            type(self).__name__,
            kwds_str(area=self.area, num_bs=self.num_bs,
                     scenario=self.traffic_model.scenario.name,
                     time=self.world_time_repr)
        )
