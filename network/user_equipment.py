from utils import *
from config import DEBUG
from . import config


class UEStatus(enum.IntEnum):
    IDLE = 0
    WAITING = 1
    ACTIVE = 2
    DONE = 3
    DROPPED = 4


class User:
    ue_height: float = config.ueHeight
    state_dim: int = 10

    def __init__(self, pos, app_type, demand, delay_budget):
        self.id = id(self)
        self.pos = np.asarray(pos)
        self.bs = None
        self.net = None
        self.app_type = app_type
        self.status = UEStatus.IDLE
        self.demand = demand
        self.delay_budget = delay_budget
        self.served = 0.
        self.delay = 0.
        self._dists = None
        self._gains = None
        self._thruput = None
        self._cover_cells = []

    # define boolean properties for each UE status
    for status in UEStatus._member_names_:
        exec(f"""@property\ndef {status.lower()}(self):
             return self.status == UEStatus.{status}""")
        
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
        if not self.active: return 0.
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
            # debug('UE {}: computed data rate {:.2f} mb/s'.format(self.id, self._thruput/1e6))
        return self._thruput
    
    @property
    def time_limit(self):
        return self.delay_budget - self.delay
    
    @property
    def required_rate(self):
        t_lim = self.time_limit
        if t_lim <= 0: return 0.
        return self.demand / t_lim
    
    @property
    def throughput_ratio(self):
        if self.required_rate <= 0: return 1.
        return min(self.data_rate / self.required_rate, 10.)
    
    @property
    def urgent(self):
        return self.time_limit < 0.03 and self.throughput_ratio < 1.

    def compute_sinr(self, noise_var=config.noiseVariance):
        if self.bs is None: return 0
        return self.signal_power / (self.interference + noise_var)
    
    @timeit
    def compute_data_rate(self):
        """
        Returns:
        The data_rate of the UE in bits/s.
        """
        self._sinr = self.compute_sinr()
        if self._sinr == 0: return 0
        return self.bs.bandwidth * np.log2(1 + self._sinr)
    
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
        # debug(f'UE {self.id} disconnects from BS {self.bs.id}')
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
        self.net.remove_user(self.id)
    
    def step(self, dt):
        # debug(f'<< {self}')
        self.delay += dt
        # if self.bs is None:
        #     self.request_connection()
        if self.active:
            size = self.data_rate * dt
            self.demand -= size
            self.served += size
        if self.demand <= 0 or self.delay >= self.delay_budget:
            self.quit()
        # debug(f'>> {self}')

    @timeit
    def info_dict(self):
        return dict(
            bs_id=self.bs.id if self.bs is not None else -1,
            status=self.status,
            demand=self.demand / 1e3,   # in kb
            deadline=self.time_limit * 1e3,  # in ms
            data_rate=self.data_rate / 1e6,  # in mb/s
            thrp_ratio=self.throughput_ratio,
            urgent=self.urgent
        )

    def get_state(self):
        return np.array(list(self.get_state_dict().values()))
    
    def __repr__(self):
        if not DEBUG:
            return 'UE(%d)' % self.id
        return 'UE({})'.format(kwds_str(
            id=self.id,
            # pos=self.pos,
            **self.info_dict()
        ))

