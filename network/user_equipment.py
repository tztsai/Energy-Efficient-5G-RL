from utils import *
from config import *
from . import config
from .channel import compute_channel_gain


class UEStatus(enum.IntEnum):
    IDLE = 0
    WAITING = 1
    ACTIVE = 2
    DONE = 3
    DROPPED = 4


class UserEquipment:
    height: float = config.ueHeight
    signal_thresh = config.signalThreshold
    record_sinr = True
    _cache = defaultdict(dict)

    def __init__(self, pos, service, demand, delay_budget):
        self.id = id(self)
        self.pos = np.asarray(pos)
        self.bs = None
        self.net = None
        self.service = service
        self.status = UEStatus.IDLE
        self.demand = self.total_demand = demand
        self.delay_budget = delay_budget
        self.delay = 0.
        self.t_served = 0.
        self._dists = None
        self._gains = None
        self._thruput = None
        self._cover_cells = []
        self._sinr_stats = Counter()

    # define boolean properties for each UE status
    for status in UEStatus._member_names_:
        exec(f"""@property\ndef {status.lower()}(self):
             return self.status == UEStatus.{status}""")

    def cached_property(func, _C=_cache):
        @property
        @wraps(func)
        def wrapped(self):
            cache = _C[self.id]
            if self.delay != cache.get('_t'):
                cache.clear()
                cache['_t'] = self.delay
            else:
                return cache[None]
            ret = func(self)
            cache[None] = ret
            return ret
        return wrapped

    @property
    def distances(self):
        return self._dists
    
    @distances.setter
    def distances(self, dists):
        self._dists = dists
        return dists

    @property
    def channel_gains(self):
        return self._gains
    
    @channel_gains.setter
    def channel_gains(self, gains):
        self._gains = gains
        q = []
        for i, bs in self.net.bss.items():
            M, K, p = bs.num_ant, bs.num_ue, bs.tx_power
            p = gains[i] * M * (M - K) * p
            if p > self.signal_thresh:
                bs.add_to_cell(self)
                q.append((p / (K + 1), i))
        self._cover_cells = [it[1] for it in sorted(q, reverse=True)]
        return gains

    @property
    def channel_gain(self):
        if self.bs is None: return 0
        return self.channel_gains[self.bs.id]

    @property
    def tx_power(self):
        if self.active:
            return self.bs.power_alloc[self.id]
        return 0.
    
    @property
    def signal_power(self):
        if not self.active:
            return 0.
        if DEBUG:
            assert self.bs.num_ant > self.bs.num_ue
        return (self.bs.num_ant - self.bs.num_ue) * self.channel_gain * self.tx_power
    
    @property
    def interference(self):
        if not self.active: return 0.
        return sum(self._gains[i] * bs.transmit_power
                   for i, bs in self.net.bss.items()
                #    for i in self._cover_cells for bs in [self.net.get_bs(i)]
                   if bs is not self.bs and not bs.sleep)
    
    @property
    def data_rate(self):
        if self._thruput is None:
            self._thruput = self.compute_data_rate()
            if DEBUG:
                debug('UE {}: computed data rate {:.2f} mb/s'.format(self.id, self._thruput/1e6))
        return self._thruput
    
    @property
    def time_limit(self):
        return self.delay_budget - self.delay
    
    @cached_property
    def required_rate(self):
        t_lim = self.time_limit
        if t_lim <= 0: return 0.
        return self.demand / t_lim
    
    @cached_property
    def throughput_ratio(self):
        if self.required_rate <= 0: return 1.
        return min(self.data_rate / self.required_rate, 10.)
    
    @property
    def urgent(self):
        return self.time_limit < 0.03

    def compute_sinr(self, N=config.noiseSpectralDensity):
        if self.bs is None: return 0
        self._S = self.signal_power
        if self.bs._has_interf:
            self._I = self.interference
            self._SINR = self._S / (self._I + N * self.bs.bandwidth)
        else:
            self._I = 0
            self._SINR = self._S / N
        if self.record_sinr:
            self._sinr_stats.update(
                T = 1,
                tx_power = self.tx_power,
                chan_gain = self.channel_gain,
                signal = self._S,
                interference = self._I,
                sinr = self._SINR
            )
        return self._SINR
    
    @timeit
    def compute_data_rate(self):
        """
        Returns:
        The data_rate of the UE in bits/s.
        """
        sinr = self.compute_sinr()
        if sinr == 0: return 0
        return self.bs.bandwidth * np.log2(1 + sinr)

    def update_data_rate(self):
        self._thruput = None
        if self.active:
            self.bs.update_power_consumption()

    # def add_to_queue(self, bs):
    #     if self._in_queue_of_bs is bs: return
    #     self.remove_from_queue()
    #     self._in_queue_of_bs = bs
    #     bs.queue[self.id] = self
    #     bs.schedule_action()
    #     info(f'UE {self.id}: added to queue of BS {bs.id}')

    # def remove_from_queue(self):
    #     if self._in_queue_of_bs:
    #         bs = self._in_queue_of_bs
    #         bs.queue.pop(self.id)
    #         self._in_queue_of_bs = None
    #         info(f'UE {self.id}: removed from queue of BS {bs.id}')

    def request_connection(self):
        assert self.idle
        for bs_id in self._cover_cells:  # in the order of channel gain
            bs = self.net.get_bs(bs_id)
            res = bs.respond_connection_request(self)
            if res: return bs_id

    def disconnect(self):
        if self.bs is None: return
        if DEBUG:
            debug(f'UE {self.id} disconnects from BS {self.bs.id}')
        if self.active:
            self.bs._disconnect(self.id)
        else:
            self.bs.pop_from_queue(self)
        self.update_data_rate()

    def quit(self):
        self.disconnect()
        for i in self._cover_cells:
            self.net.get_bs(i).remove_from_cell(self)
        if self.demand <= 0.:
            self.demand = 0.
            self.status = UEStatus.DONE
        else:
            self.status = UEStatus.DROPPED
        # self.served = self.total_demand - self.demand
        if EVAL:
            dropped = self.demand
            done = self.total_demand - dropped
            delay = self.delay
            service_time = self.t_served
            steps = self._sinr_stats.pop('T', 1)
            self.net.add_stat('ue_stats', dict(
                demand = self.total_demand,
                done = done,
                dropped = dropped,
                delay = delay,
                delay_budget = self.delay_budget,
                service_time = service_time,
                **{'avg_'+k: self._sinr_stats[k]/steps
                   for k in list(self._sinr_stats)}
            ))
        self.net.remove_user(self.id)
        del self.__class__._cache[self.id]
    
    def step(self, dt):
        # DEBUG and debug(f'<< {self}')
        self.delay += dt
        if EVAL and self.active:
            self.t_served += dt
        if self.active:
            self.demand -= self.data_rate * dt
        if self.demand <= 0 or self.delay >= self.delay_budget:
            self.quit()
        # DEBUG and debug(f'>> {self}')

    @timeit
    def info_dict(self):
        return dict(
            bs_id=self.bs.id if self.bs else '-',
            status=self.status.name,
            demand=self.demand / 1e3,   # in kb
            rate=self.compute_data_rate() / 1e6,  # in mb/s
            ddl=self.time_limit * 1e3,  # in ms
            urgent=self.urgent
        )
    
    def __repr__(self):
        return 'UE(%d)' % self.id
        return 'UE({})'.format(kwds_str(
            id=self.id,
            # pos=self.pos,
            **self.info_dict()
        ))


class TestUE(UserEquipment):
    record_sinr = False
    
    def __init__(self, net, grid_size=config.probeGridSize):
        super().__init__(None, None, 0, 999)
        self.net = net
        self.x = np.linspace(0, net.area[0], round(net.area[0] / grid_size) + 1)
        self.y = np.linspace(0, net.area[1], round(net.area[1] / grid_size) + 1)
        
    @property
    def tx_power(self):
        return self._tx_power

    def test_sinr(self):
        """ Measure SINR over the grid. """
        print('Probing SINR...')
        self.status = UEStatus.ACTIVE
        # bss = np.arange(self.net.num_bs)
        poses = np.array(list(itertools.product(self.x, self.y, [self.height])))
        csi_keys = ['S', 'I', 'SINR'] # + [f'C_{i}' for i in range(self.net.num_bs)]
        # csi_index = pd.MultiIndex.from_product([self.x, self.y, bss], names=['x', 'y', 'bs'])
        csi_index = pd.MultiIndex.from_product([self.x, self.y], names=['x', 'y'])
        csi = np.zeros((len(csi_index), len(csi_keys)))
        distances = np.sqrt(np.sum((poses[:,None] - self.net.bs_positions)**2, axis=-1))
        channel_gains = compute_channel_gain(distances)
        N = config.noiseSpectralDensity
        i = 0
        for dists, gains in zip(distances, channel_gains):
            self.distances = dists
            self.channel_gains = gains
            for c in self._cover_cells:
                bs = self.net.get_bs(c)
                if not self.bs and bs.responding:
                    self.bs = bs
                    # G = self.channel_gain
                    if bs.sleep or bs.ues_full:
                        S = I = 0
                    else:
                        self._tx_power = bs.transmit_power / (bs.num_ue + 1)
                        S = self.signal_power
                        I = self.interference
                    SINR = S / (I + N)
                bs.remove_from_cell(self)
            if not self.bs:
                S = I = SINR = 0
            self.bs = None
            cover_cells = [0] * self.net.num_bs
            for c in self._cover_cells:
                cover_cells[c] = 1
            csi[i] = list(map(lin2dB, [S, I, SINR]))
            i += 1
        return pd.DataFrame(csi, index=csi_index, columns=csi_keys)
