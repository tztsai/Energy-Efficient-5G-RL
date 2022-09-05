from utils import *
from env.utils import *
from . import config
from .user_equipment import User, UEStatus
from .base_station import BaseStation
from .channel import compute_channel_gain
from traffic import TrafficModel
from config import DEBUG


class Green5GNet:
    inter_bs_dist = config.interBSDist
    default_area = config.areaSize
    default_bs_poses = config.bsPositions
    
    global_obs_space = make_box_env([[0, np.inf]] * (1 + 3 + 3 + 2))
    bs_obs_space = concat_box_envs(
        BaseStation.self_obs_space,
        duplicate_box_env(
            concat_box_envs(
                BaseStation.public_obs_space,
                BaseStation.mutual_obs_space),
            config.numBS - 1))
    net_obs_space = concat_box_envs(
        global_obs_space,
        duplicate_box_env(BaseStation.self_obs_space, config.numBS),
        duplicate_box_env(BaseStation.mutual_obs_space,
                          config.numBS * (config.numBS - 1) // 2))

    bs_obs_ndims = box_env_ndims(bs_obs_space)
    net_obs_ndims = box_env_ndims(net_obs_space)
    
    def __init__(self, area, bs_poses, traffic_type, start_time=0, accel_rate=1):
        self.area = area
        self.traffic_model = TrafficModel.from_scenario(traffic_type)
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
        return int(d + 1), int(h), int(m), s

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
    def new_demand_rates(self):
        if self._timer:
            return self._demand / self._timer / 1e6
        return np.zeros_like(self._demand)
    
    @property
    def drop_rates(self):
        """ Drop rates for each app category in the network in kb/s. """
        if self._timer:
            return self._dropped / self._timer / 1e3
        return np.zeros_like(self._dropped)

    def get_bs(self, id):
        return self.bss[id]

    def get_ue(self, id):
        return self.ues[id]

    def add_base_station(self, bs):
        # assert bs.id not in self.bss, "BS %d already in the network" % bs.id
        bs.net = self
        self.bss[bs.id] = bs
        info(f'{bs} added to the network')

    def create_new_bs(self, id, pos, **kwargs):
        pos = np.append(pos, BaseStation.bs_height)
        bs = BaseStation(id, pos, net=self, **kwargs)
        self.add_base_station(bs)
        return bs

    def add_user(self, ue):
        # assert ue.id not in self.ues, "UE %s already in the network" % ue.id
        ue.net = self
        self.ues[ue.id] = ue
        self.measure_distances_and_gains(ue)
        # debug(f'UE({ue.id}, cat={ue.app_type}) added to the network')

    @timeit
    def scan_connections(self):
        for ue in self.ues.values():
            if ue.idle:
                ue.request_connection()

    @timeit
    def measure_distances_and_gains(self, ue):
        ue.distances = np.sqrt(np.sum((self.bs_positions - ue.pos)**2, axis=1))
        # debug(f'Distances of UE {ue.id} to BSs: {ue.dists}')
        dists = [ue.distances[i] for i in ue._cover_cells]
        gains = compute_channel_gain(dists)
        ue.channel_gains = {i: g for i, g in zip(ue._cover_cells, gains)}

    def remove_user(self, ue_id):
        assert ue_id in self.ues
        ue = self.ues[ue_id]
        del self.ues[ue_id]
        if ue.done:
            self._total_done[ue.app_type] += [1, ue.served/1e6, ue.delay]
            # debug(f"{ue} done")
        else:
            if ue.demand <= 0: breakpoint()
            self._dropped[ue.app_type] += ue.demand / 1e6
            self._total_dropped[ue.app_type] += [1, ue.demand/1e6, ue.delay]
            debug(f"{ue} dropped")

    @timeit
    def generate_new_ues(self, dt, **kwargs):
        new_traffic = self.traffic_model.emit_traffic(self.world_time, dt)
        for app, (demand, delay) in enumerate(new_traffic):
            if not demand: continue
            if 'pos' not in kwargs:
                xy = np.random.rand(2) * self.area
                kwargs['pos'] = np.append(xy, User.ue_height)
            ue = User(app_type=app, demand=demand, delay_budget=delay, **kwargs)
            self._demand[app] += demand
            self.add_user(ue)

    def consume_energy(self, energy):
        self._energy_consumed += energy

    def reset(self):
        info('Resetting %s', self)
        for bs in self.bss.values():
            bs.reset()
        self.ues.clear()
        self._time = 0
        self._demand = np.zeros(self.traffic_model.num_apps)
        self._dropped = np.zeros(self.traffic_model.num_apps)
        self._total_dropped = np.zeros((self.traffic_model.num_apps, 3))
        self._total_done = np.zeros((self.traffic_model.num_apps, 3))
        self._energy_consumed = 0
        self._total_energy_consumed = 0
        self._timer = 0
        self.reset_stats()

    @timeit
    def step(self, dt):
        self.generate_new_ues(dt)
    
        self.scan_connections()

        for bs in self.bss.values():
            bs.step(dt)

        for ue in list(self.ues.values()):
            ue.step(dt)

        self._total_energy_consumed += self._energy_consumed
        self._timer += dt
        self._time += dt

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
                bs_obs.append(self.bss[i].observe_other(self.bss[j]))
        bs_obs = np.concatenate(bs_obs, dtype=np.float32)
        thrp_ratio, thrp_req = BaseStation.calc_sum_rate(self.ues.values())
        return np.concatenate([
            [self.power_consumption],   # power consumption (1)
            self.new_demand_rates,      # rates demanded by new UEs in different delay cats (3)
            self.drop_rates,            # dropped rates in different delay cats (3)
            [thrp_ratio, thrp_req],     # throughput (2)
            bs_obs                      # bs observations
        ], dtype=np.float32)

    def reset_stats(self):
        self._demand[:] = 0
        self._dropped[:] = 0
        self._energy_consumed = 0
        self._timer = 0
        for bs in self.bss.values():
            bs.reset_stats()

    @cache_obs
    def info_dict(self):
        obs = self.observe_network()
        thrp_ratio, thrp_req = obs[5:7]
        return dict(
            time='Day {}, {:02}:{:02}:{:02.2f}'.format(*self.world_time_tuple),
            power_consumption=obs[0],
            new_demand_rate=obs[1:4].sum(),
            dropped_rate=obs[4:7].sum(),
            required_rate=thrp_req,
            throughput=thrp_ratio * thrp_req,
            avg_pc=self.avg_power_consumption,
            avg_data_rates=self._total_done[:, 1] /
                np.maximum(self._total_done[:, 2], 1e-6),
            avg_drop_rates=self._total_dropped[:, 1] /
                np.maximum(self._total_dropped[:, 2], 1e-6),
            total_done_count=self._total_done[:, 0].sum(),
            total_dropped_count=self._total_dropped[:, 0].sum(),
            total_drop_ratios=self._total_dropped[:, 1] / 
                (self._total_done[:, 1] + self._total_dropped[:, 1] + 1e-6),
            # bs_info=pd.DataFrame({i: bs.info_dict() for i, bs in self.bss.items()}).T,
            # ue_info=pd.DataFrame({i: ue.info_dict() for i, ue in self.ues.items()}).T
        )
        
    def __repr__(self) -> str:
        if not DEBUG:
            return 'Network'
        return '{}({})'.format(
            type(self).__name__,
            kwds_str(area=self.area, num_bs=self.num_bs,
                     scenario=self.traffic_model.scenario,
                     start_time=self.traffic_model.get_time_slot(self.start_time))
        )
