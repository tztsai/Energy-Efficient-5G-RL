from gym.spaces import Box
from collections import deque
from utils import *
from . import config
from .env_utils import *
from .user_equipment import User, UEStatus
# from visualize.obs import VisBSStats
from config import DEBUG


class ConnectMode(enum.IntEnum):
    Disconnect = -1
    Reject = 0
    Accept = 1


class BaseStation:
    num_antennas = config.numAntennas
    ant_power = config.antennaPower
    band_width = config.bandWidth
    frequency = config.bsFrequency
    bs_height = config.bsHeight
    cell_radius = config.cellRadius
    num_conn_modes = len(ConnectMode)
    num_sleep_modes = len(config.sleepDiscounts)
    num_ant_switch_opts = len(config.antennaSwitchOpts)
    wakeup_delays = config.wakeupDelays
    ant_switch_opts = config.antennaSwitchOpts
    ant_switch_energy = config.antSwitchEnergy
    sleep_switch_energy = config.sleepSwitchEnergy
    disconnect_energy = config.disconnectEnergy
    power_alloc_weights = config.powerAllocWeights
    add_pc_penalty = config.addPCPenalty
    buffer_size = (60, 2)
    buffer_chunk_size = 5
    buffer_num_chunks = buffer_size[0] // buffer_chunk_size
    bs_stats_dim = buffer_num_chunks * buffer_size[1]
    ue_stats_dim = 11  #10
    
    public_obs_space = make_box_env(
        [[0, num_antennas], [0, 1]] +
        [[0, 1]] * num_sleep_modes
    )
    private_obs_space = make_box_env(
        # [[0, 6], [0, 59], [0, 60]] +    # time
        [[0, 1]] * num_sleep_modes +    # next sleep mode
        [[0, 99]] +                     # wakeup time
        [[0, np.inf]] * bs_stats_dim +  # bs stats
        [[0, np.inf]] * ue_stats_dim    # ue stats
    )
    mutual_obs_space = make_box_env([[0, np.inf]] * 6)
    self_obs_space = concat_box_envs(public_obs_space, private_obs_space)
    other_obs_space = concat_box_envs(public_obs_space, mutual_obs_space)
    
    public_obs_ndims = box_env_ndims(public_obs_space)
    private_obs_ndims = box_env_ndims(private_obs_space)
    self_obs_ndims = box_env_ndims(self_obs_space)
    other_obs_ndims = box_env_ndims(other_obs_space)
    
    action_dims = (num_ant_switch_opts, num_sleep_modes, num_conn_modes)

    def __init__(
        self, id, pos, net, 
        ant_power=None, num_antennas=None,
        frequency=None, band_width=None,
    ):
        pos = np.asarray(pos)
        for k, v in locals().items():
            if v is not None and k != 'self':
                setattr(self, k, v)
        self.ues = dict()
        self.queue = deque()
        self.covered_ues = set()
        self._nb_dists = dict()
        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._buf_idx = 0
        self.reset()

    ### properties ###
    
    @property
    def num_ue(self):
        return len(self.ues)
    
    @property
    def ues_full(self):
        return self.num_ue >= self.num_ant - 1
    
    @property
    def responding(self):
        return self.conn_mode > 0
    
    @property
    def transmit_power(self):
        return 0 if self.sleep else self.ant_power * self.num_ant
    
    @property
    def sum_rate(self):
        return sum(ue.data_rate for ue in self.ues.values())

    @property
    def power_alloc(self):
        if self._power_alloc is None:
            self.alloc_power()
        return self._power_alloc

    @property
    def operation_pc(self):  # operation power consumption
        if self._pc is None:
            self._pc = self.compute_power_consumption()
        return self._pc
    
    @property
    def power_consumption(self):
        ec = sum(v for k, v in self._energy_consumed.items() if
                 self.add_pc_penalty or k == 'operation')
        return self._timer and ec / self._timer
    
    @property
    def cell_traffic_rate(self):
        return self._steps and self._arrival_rate / self._steps
    
    @property
    def wakeup_time(self):
        if self.sleep == self._next_sleep:
            return 0.
        else:
            return self._wake_delay - self._wake_timer
    
    ### actions ###
    
    @timeit
    def take_action(self, action):
        assert len(action) == len(self.action_dims)
        # info(f'BS {self.id} takes action:\n{action}')
        self.switch_antennae(int(action[0]))
        self.switch_sleep_mode(int(action[1]))
        self.switch_connection_mode(int(action[2]))
    
    def switch_antennae(self, opt):
        assert opt in range(self.num_ant_switch_opts)
        num_switch = self.ant_switch_opts[opt]
        if num_switch == 0: return
        energy_cost = self.ant_switch_energy * abs(num_switch)
        self.consume_energy(energy_cost, 'antenna')
        num_ant_new = self.num_antennas + num_switch
        if num_ant_new <= self.num_ue or num_ant_new > self.num_antennas:
            return  # invalid action
        self.num_ant = num_ant_new
        for ue in self.ues.values():
            ue.update_data_rate()
        info(f'BS {self.id}: switched to {self.num_ant} antennas')
        self.update_power_allocation()
        self.update_power_consumption()
    
    def switch_sleep_mode(self, mode):
        assert mode in range(self.num_sleep_modes)
        if mode == self.sleep:
            self._next_sleep = mode
            return
        self.consume_energy(self.sleep_switch_energy[mode], 'sleep')
        if mode == 3 and any(ue.status < 2 for ue in self.covered_ues):
            return  # cannot go to deep sleep if there are inactive UEs in coverage
        self._next_sleep = mode
        if mode > self.sleep:
            info('BS {}: goes to sleep {} -> {}'.format(self.id, self.sleep, mode))
            self._prev_sleep = self.sleep
            self.sleep = mode
        elif mode < self.sleep:
            self._wake_delay = self.wakeup_delays[self.sleep]

    def switch_connection_mode(self, mode):
        """
        Mode 0: disconnect all UEs and refuse new connections
        Mode 1: refuse new connections
        Mode 2: accept new connections
        Mode 3: accept new connections and take over all UEs in cell range
        """
        assert mode in range(self.num_conn_modes)
        mode -= 1  # -1, 0, 1
        self.conn_mode = mode
        if self.conn_mode > 0 and self.sleep == 3:  # cannot accept new connections in SM3
            self.consume_energy(2, 'connect')
            self.conn_mode = -1
        if self.conn_mode < 0:  # disconnect all ues and empty the queue
            self.disconnect_all()
        # elif mode == 3:  # take over all ues
        #     if self.sleep:  # cannot take over UEs if asleep
        #         self.consume_energy(2, 'connect')  # add EC penalty
        #     else:
        #         self.takeover_all()
    
    ### network functions ###
    
    def neighbor_dist(self, bs_id):
        if bs_id in self._nb_dists:
            return self._nb_dists[bs_id]
        bs = self.net.get_bs(bs_id)
        d = np.linalg.norm(self.pos - bs.pos) / 1000  # km
        self._nb_dists[bs_id] = d
        bs._nb_dists[self.id] = d
        return d

    def connect(self, ue):
        assert ue.bs is None
        self.ues[ue.id] = ue
        ue.bs = self
        ue.status = UEStatus.ACTIVE
        self.update_power_allocation()
        self.update_power_consumption()
        # debug('BS {}: connected UE {}'.format(self.id, ue.id))

    def _disconnect(self, ue_id):
        """ Don't call this directly. Use UE.disconnect() instead. """
        ue = self.ues.pop(ue_id)
        ue.bs = None
        ue.status = UEStatus.IDLE
        self.update_power_allocation()
        self.update_power_consumption()
        # debug('BS {}: disconnected UE {}'.format(self.id, ue_id))

    def respond_connection_request(self, ue):
        if self.responding:
            if ue.idle:
                if self.sleep or self.ues_full:
                    self.add_to_queue(ue)
                else:
                    self.connect(ue)
            else:
                breakpoint()
            return True

    def add_to_cell(self, ue):
        self.covered_ues.add(ue)
        self._arrival_rate += ue.required_rate / 1e6
    
    def remove_from_cell(self, ue):
        self.covered_ues.remove(ue)

    def takeover_all(self):
        if self.covered_ues:
            info(f'BS {self.id}: takes over all UEs in cell')
        for ue in self.covered_ues:
            if ue.bs is not self:
                if ue.bs is not None:
                    ue.disconnect()
                    self.consume_energy(self.disconnect_energy, 'disconnect')
                self.add_to_queue(ue)  # delay connection to the next step

    def add_to_queue(self, ue):
        assert ue.idle
        self.queue.append(ue)
        ue.bs = self
        ue.status = UEStatus.WAITING
        # debug('BS {}: added UE {} to queue'.format(self.id, ue.id))
        
    def pop_from_queue(self, ue=None):
        if ue is None:
            ue = self.queue.popleft()
        else:
            self.queue.remove(ue)
        ue.bs = None
        ue.status = UEStatus.IDLE
        # debug('BS {}: removed UE {} from queue'.format(self.id, ue.id))
        return ue
    
    ### state transition ###

    def update_power_allocation(self):
        self._power_alloc = None

    def update_power_consumption(self):
        self._pc = None

    @timeit
    def alloc_power(self):
        if not self.ues: return
        w = np.array([self.power_alloc_weights[ue.app_type]
                      for ue in self.ues.values()])
        ps = self.transmit_power * w / w.sum()
        self._power_alloc = dict(zip(self.ues.keys(), ps))
        for ue in self.net.ues.values():
            ue.update_data_rate()
        # info('BS {}: allocated power {}'.format(self.id, self._power_alloc))

    @timeit
    def update_sleep(self, dt):
        if self._next_sleep == self.sleep:
            if self.queue and self.sleep in (1, 2):
                # info('BS {}: automatically waking up'.format(self.id))
                self.switch_sleep_mode(0)
            elif self.sleep == 0 and not self.ues and self._prev_sleep == 1:
                self.switch_sleep_mode(1)
                info('BS {}: automatically goes to sleep mode 1'.format(self.id))
            return
        self._wake_timer += dt
        if self._wake_timer >= self._wake_delay:
            info('BS {}: switched sleep mode {} -> {}'
                  .format(self.id, self.sleep, self._next_sleep))
            self.consume_energy(self.sleep_switch_energy[self.sleep], 'wakeup')
            self._prev_sleep = self.sleep
            self.sleep = self._next_sleep
            self._wake_timer = 0.
        # else:
        #     wake_time = (self._wake_delay - self._wake_timer) * 1000
        #     info('BS {}: switching sleep mode {} -> {} (after {:.0f} ms)'
        #          .format(self.id, self.sleep, self._next_sleep, wake_time))

    @timeit
    def update_connections(self):
        if self.sleep:  # move all current ues to the queue
            for ue in list(self.ues.values()):
                ue.disconnect()
                self.add_to_queue(ue)
        else:  # connect ues in the queue
            while self.queue and not self.ues_full:
                ue = self.pop_from_queue()
                self.connect(ue)

    def disconnect_all(self):
        if self.ues or self.queue:
            info('BS {}: disconnects all UEs'.format(self.id))
        for ue in list(self.ues.values()):
            ue.disconnect()
            self.consume_energy(self.disconnect_energy, 'disconnect')
        while self.queue:
            self.pop_from_queue()

    @timeit
    def compute_power_consumption(
        self, eta=0.8, eps=8.2e-3, Ppa_max=40, Psyn=2,
        Pbs=1, Pcd=0.9, Lbs=12.8, Tc=1800, Poth=18, C={},
        sm_discounts=config.sleepDiscounts
    ):
        """
        Args:
        - eta: max PA efficiency of the BS
        - Ppa_max: max PA power consumption
        - Psyn: sync power
        - Pbs: power consumption of circuit components
        - Pcd: power consumption of coding/decoding
        
        Returns:
        The power consumption of the BS in Watts.
        """
        M = self.num_ant
        K = self.num_ue
        if 3 not in C:
            B = self.band_width / 1e9
            C[3] = B / (3 * Tc * Lbs)
            C[11] = B / Lbs * (2 + 1/Tc)
            C[12] = 3 * B / Lbs
            C['ET-PA'] = (self.ant_power + eps*Ppa_max) / ((1+eps)*eta)
        if self.sleep:
            Ppa = 0
            Pbb = M * Pbs * sm_discounts[self.sleep]
        else:  # assume ET-PA (envelope tracking power amplifier)
            Ppa = M / C['ET-PA']
            Pbb = Psyn + M * Pbs
            if K > 0:
                R = sum(ue.data_rate for ue in self.ues.values()) / 1e6
                Pbb += Pcd * R + C[3]*K**3 + M * (C[11]*K + C[12]*K**2)
        p = Ppa + Pbb + Poth
        # debug(f'BS {self.id}: Ppa={Ppa}, Pbb={Pbb}, P={p}')
        return p
    
    def consume_energy(self, e, k):
        self._energy_consumed[k] += e
        self.net.consume_energy(e)

    def insert_buffer(self, record):
        self._buffer[self._buf_idx] = record
        self._buf_idx = (self._buf_idx + 1) % len(self._buffer)

    ### called by the environment ###
    
    def reset(self):
        self.ues.clear()
        self.queue.clear()
        self.covered_ues.clear()
        self.sleep = 0
        self.conn_mode = 1
        self.num_ant = self.num_antennas
        self._power_alloc = None
        self._prev_sleep = 0
        self._next_sleep = 0
        self._pc = None
        self._time = 0
        self._timer = 0
        self._steps = 0
        self._wake_timer = 0
        self._wake_delay = 0
        self._energy_consumed = defaultdict(float)
        self._total_energy_consumed = 0

    def step(self, dt):
        self.update_sleep(dt)
        self.update_connections()
        self.consume_energy(self.operation_pc * dt, 'operation')
        self.update_timer(dt)

    @timeit
    @cache_obs
    def get_observation(self):
        obs = [self.observe_self()]
        for bs in self.net.bss.values():
            if bs is self: continue
            obs.append(bs.observe_self()[:self.public_obs_ndims])
            obs.append(self.observe_other(bs)[0])
        return np.concatenate(obs, dtype=np.float32)

    @timeit
    @cache_obs
    def observe_self(self):
        # hour, sec = divmod(self.net.world_time, 3600)
        # day, hour = divmod(self.net.world_time, 24)
        return np.concatenate([
            ### public information ###
            # self.pos
            # [self.band_width, self.transmit_power],
            [self.num_ant, self.responding],
            onehot_vec(self.num_sleep_modes, self.sleep),
            ### private information ###
            # [day % 7 + 1, hour, sec],
            onehot_vec(self.num_sleep_modes, self._next_sleep),
            [self.wakeup_time],
            self.get_bs_stats(),
            self.get_ue_stats()
        ], dtype=np.float32)

    @timeit
    @cache_obs
    def observe_other(self, bs):
        shared_ues = self.covered_ues & bs.covered_ues
        owned_ues = set(ue for ue in shared_ues if ue.bs is self)
        others_ues = set(ue for ue in shared_ues if ue.bs is bs)
        owned_log_ratio = self.calc_sum_rate(owned_ues)[-1]
        others_log_ratio = self.calc_sum_rate(others_ues)[-1]
        obs = np.array([
            self.neighbor_dist(bs.id),
            len(shared_ues), len(owned_ues), len(others_ues),
            owned_log_ratio, others_log_ratio
        ], dtype=np.float32)
        other_obs = obs[[0, 1, 3, 2, 5, 4]]
        return obs, other_obs
    
    @staticmethod
    def calc_sum_rate(ues):
        required_thrp = sum(ue.required_rate for ue in ues)
        real_thrp = sum(ue.data_rate for ue in ues)
        if required_thrp > 0:
            required_thrp *= 1e-6
            if real_thrp == 0:
                return real_thrp, required_thrp, -10.
            real_thrp *= 1e-6
            ratio = np.clip(real_thrp / required_thrp, 1e-4, 1e4)
            return real_thrp, required_thrp, np.log(ratio)
        else:
            real_thrp *= 1e-6
            return real_thrp, 0, 0

    # @VisBSStats
    def get_bs_stats(self):
        idx = [(self._buf_idx + i * self.buffer_chunk_size) % len(self._buffer)
               for i in range(self.buffer_num_chunks + 1)]
        chunks = np.array([self._buffer[i:j] if i < j else
                           np.vstack([self._buffer[i:], self._buffer[:j]])
                           for i, j in zip(idx[:-1], idx[1:])], dtype=np.float32)
        # yield chunks
        return chunks.mean(axis=1).flatten()

    def get_ue_stats(self):
        idle_ues, others_ues = [], []
        for ue in self.covered_ues:
            if ue.bs is None:
                idle_ues.append(ue)
            elif ue.bs is not self:
                others_ues.append(ue)
        thrp, thrp_req, log_ratio = self.calc_sum_rate(self.ues.values())
        thrp_req_queued = self.calc_sum_rate(self.queue)[1]
        thrp_req_idle = self.calc_sum_rate(idle_ues)[1]
        # others_ues += idle_ues
        thrp_other, thrp_req_other, log_ratio_other = self.calc_sum_rate(others_ues)
        return [
            len(self.ues), len(self.queue), len(self.covered_ues),
            thrp,
            thrp_other,
            thrp_req, 
            thrp_req_queued, 
            thrp_req_idle, 
            thrp_req_other,
            log_ratio,
            log_ratio_other
        ]

    def update_timer(self, dt):
        self._steps += 1
        self._time += dt
        self._timer += dt

    def update_stats(self):
        # debug(f'BS {self.id}: energy consumption {kwds_str(**self._energy_consumed)}')
        record = [self.power_consumption, self.cell_traffic_rate]
        # debug(f'BS {self.id}: power consumption {record[0]} W')
        self.insert_buffer(record)
        self._total_energy_consumed += record[0] * self._timer

    def reset_stats(self):
        self._steps = 0
        self._timer = 0
        self._arrival_rate = 0
        self._energy_consumed.clear()
        if self.conn_mode < 0:
            self.conn_mode = 0

    @cache_obs
    def info_dict(self):
        obs = self.observe_self()
        return dict(
            num_ant=self.num_ant,
            conn_mode=self.conn_mode,
            sleep=self.sleep,
            next_sleep=self._next_sleep,
            wakeup=self.wakeup_time,
            pc=self.power_consumption,
            num_s=len(self.ues),
            num_q=len(self.queue),
            num_c=len(self.covered_ues),
            thrp=obs[-7],
            thrp_req=obs[-5],
            thrp_req_q=obs[-4],
            thrp_req_o=obs[-3],
            thrp_ratio_o=np.exp(obs[-1])
        )

    def __repr__(self):
        if not DEBUG:
            return 'BS(%d)' % self.id
        return 'BS({})'.format(kwds_str(
            id=self.id,
            pos=self.pos,
            **self.info_dict()
        ))
