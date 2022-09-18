from utils import *
from config import *
from . import config


class UEStatus(enum.IntEnum):
    IDLE = 0
    WAITING = 1
    ACTIVE = 2
    DONE = 3
    DROPPED = 4


class UserEquipment:
    height: float = config.ueHeight
    state_dim: int = 10
    _cache = defaultdict(dict)

    def __init__(self, pos, app_type, demand, delay_budget):
        self.id = id(self)
        self.pos = np.asarray(pos)
        self.bs = None
        self.net = None
        self.app_type = app_type
        self.status = UEStatus.IDLE
        self.demand = self.file_size = demand
        self.delay_budget = delay_budget
        self.delay = 0.
        self.t_served = 0.
        self._dists = None
        self._gains = None
        self._thruput = None
        self._cover_cells = []

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
        assert self._dists is not None, 'distances to BSs not computed yet'
        return self._dists
    
    @distances.setter
    def distances(self, dists):
        assert self._dists is None
        self._dists = dists
        self._cover_cells = []
        for i, bs in self.net.bss.items():
            if dists[i] <= bs.cell_radius:
                self._cover_cells.append(bs.id)
                bs.add_to_cell(self)
        return dists

    @property
    def channel_gains(self):
        return self._gains
    
    @channel_gains.setter
    def channel_gains(self, gains):
        self._gains = gains
        self._cover_cells.sort(key=gains.__getitem__, reverse=True)
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

    def compute_sinr(self, N=config.noisePower):
        if self.bs is None: return 0
        S = self.signal_power
        I = self.interference
        SINR = S / (I + N)
        if EVAL:
            rec = dict(
                x = self.pos[0],
                y = self.pos[1],
                bs = self.bs.id,
                p = self.tx_power,
                M = self.bs.num_ant,
                K = self.bs.num_ue,
                d = self._dists[self.bs.id],
                g = self.channel_gain,
                S = S, I = I, SINR = SINR,
            )
            for i, bs in self.net.bss.items():
                rec[f'P_{i}'] = bs.transmit_power
            self.net.add_stat('sinr', rec)
        return SINR
    
    @timeit
    def compute_data_rate(self):
        """
        Returns:
        The data_rate of the UE in bits/s.
        """
        self.sinr = self.compute_sinr()
        if self.sinr == 0: return 0
        return self.bs.bandwidth * np.log2(1 + self.sinr)

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
        if self.demand <= 0:
            self.status = UEStatus.DONE
        else:
            self.status = UEStatus.DROPPED
        # self.served = self.total_demand - self.demand
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
            bs_id=self.bs.id if self.bs is not None else -1,
            status=self.status,
            demand=self.demand / 1e3,   # in kb
            thrp=self.compute_data_rate() / 1e6,  # in mb/s
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

