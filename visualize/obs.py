import numpy as np
import matplotlib.pyplot as plt
from utils import trace_locals, wraps


def anim_rolling(func):
    fig, ax = plt.subplots()
    func = trace_locals(func)
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            ax.cla()
            self = func['self']
            y = self._arrival_buf[[(self._buf_idx + i + 1) % self.buffer_ws
                                   for i in range(self.buffer_ws)]]
            ax.plot(y)
            plt.pause(0.001)
        elif self.id == 0:
            chunks = func['chunks']
            arr = np.concatenate(chunks)
            # plot chunks
            ax.cla()
            ax.plot(arr[:,0] / 100, label='pc')
            ax.plot(arr[:,1], label='traffic')
            avg_arr = np.mean(chunks, axis=1)
            xx = np.arange(0, arr.shape[0], chunks.shape[1])
            ax.plot(xx, avg_arr[:,0] / 100, label='avg_pc')
            ax.plot(xx, avg_arr[:,1], label='avg_traffic')
            ax.set_title('time: ' + str(self._time))
            ax.legend()
            plt.pause(0.001)
        return ret
    return wrapped
