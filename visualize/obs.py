import numpy as np
import matplotlib.pyplot as plt
from utils import trace_locals


class VisRolling:
    fig, ax = plt.subplots()
    
    def __new__(cls, func):
        func = trace_locals(func)
        def wrapped(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            if not hasattr(self, 'id'):
                cls.ax.cla()
                self = func['self']
                y = self._arrival_buf[[(self._buf_idx + i + 1) % self.buffer_ws
                                       for i in range(self.buffer_ws)]]
                cls.ax.plot(y)
                plt.pause(0.001)
            elif self.id == 0:
                chunks = func['chunks']
                arr = np.concatenate(chunks)
                # plot chunks
                cls.ax.cla()
                cls.ax.plot(arr[:,0] / 100, label='pc')
                cls.ax.plot(arr[:,1], label='traffic')
                avg_arr = np.mean(chunks, axis=1)
                xx = np.arange(0, arr.shape[0], chunks.shape[1])
                cls.ax.plot(xx, avg_arr[:,0] / 100, label='avg_pc')
                cls.ax.plot(xx, avg_arr[:,1], label='avg_traffic')
                cls.ax.set_title('time: ' + str(self._time))
                cls.ax.legend()
                plt.pause(0.001)
            return ret
        return wrapped
