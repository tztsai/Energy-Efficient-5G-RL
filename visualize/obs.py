import numpy as np
import matplotlib.pyplot as plt

class VisBSStats:
    fig, ax = plt.subplots()
    
    def __new__(cls, func):
        def wrapped(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            chunks = next(ret)
            arr = np.concatenate(chunks)
            if self.id == 0:
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
            try:
                next(ret)
            except StopIteration as e:
                return e.value
        return wrapped
