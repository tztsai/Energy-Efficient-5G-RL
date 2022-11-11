# %%
import numpy as np
import tensorflow as tf
from functools import partial
from tensorflow import keras
from tensorflow.keras.layers import *
from keras.utils.vis_utils import plot_model

# %%
class MLP(keras.Sequential):
    def __init__(self) -> None:
        super().__init__()
        self.add(Normalization(axis=-1))
        self.add(Dense(64, activation='relu'))
        self.add(LayerNormalization(axis=-1))
        self.add(Dense(64, activation='relu'))
        self.add(LayerNormalization(axis=-1))
        self.add(Dense(64, activation='relu'))
        self.add(LayerNormalization(axis=-1))
        
class Policy(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return
    
class Environment(keras.layers.Layer):
    def __init__(self, *args) -> None:
        super().__init__()
        
    def call(self, inputs):
        return
    
class Reward(keras.layers.Layer):
    def __init__(self, *args) -> None:
        super().__init__()
        
    def call(self, inputs):
        return
    
class Advantage(keras.layers.Layer):
    def __init__(self, *args) -> None:
        super().__init__()
        
    def call(self, inputs):
        return

class PolicyLoss(keras.layers.Layer):
    def __init__(self, *args) -> None:
        super().__init__()
    
    def call(self, inputs):
        return
        
class ValueLoss(keras.layers.Layer):
    def __init__(self, *args) -> None:
        super().__init__()
        
    def call(self, inputs):
        return
    
class Observations(InputLayer):
    def __init__(self, *args) -> None:
        super().__init__(*args)

class State(InputLayer):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        
# Policy = ValueLoss = PolicyLoss = partial(Lambda, lambda x: x)

# %%
obs = Input((7, 116))
x = Normalization(axis=-1)(obs)
x = Dense(64, activation="relu")(x)
x = LayerNormalization(axis=-1)(x)
x = Dense(64, activation="relu")(x)
x = LayerNormalization(axis=-1)(x)
x = Dense(64, activation="relu")(x)
act_in = LayerNormalization(axis=-1)(x)
act1 = Dense(3, activation="softmax")(act_in)
act2 = Dense(4, activation="softmax")(act_in)
act3 = Dense(3, activation="softmax")(act_in)
# policy = Policy()([act1, act2, act3])
policy = [act1, act2, act3]
actor = keras.Model(inputs=obs, outputs=policy, name="actor")
actor.summary()

from keras_flops import get_flops
flops = get_flops(actor, batch_size=1)
# print(f"FLOPS: {flops / 10 ** 6:.03} M")

# %%
import torch
from torch import nn

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.LayerNorm(116),
            nn.Linear(116, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.a1 = nn.Linear(64, 3)
        self.a2 = nn.Linear(64, 4)
        self.a3 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = self.base(x)
        a1 = self.a1(x).argmax(dim=-1)
        a2 = self.a2(x).argmax(dim=-1)
        a3 = self.a3(x).argmax(dim=-1)
        return a1#, a2, a3

from pthflops import count_ops
model = TorchModel()
tot_flops, op_flops = count_ops(model, torch.ones(116))
flops_df = pd.DataFrame(op_flops, columns=['layer', 'single pass ops']).set_index('layer')
flops_df['Gflops'] = flops_df['single pass ops'] * 7 * 50 / 1e9
flops_df['pc (mW)'] = flops_df['Gflops'] * 1000 / 12.8
flops_df.loc['TOTAL'] = flops_df.sum(axis=0)
flops_df
        
# %%
state = Input(shape=(821,))
x = Normalization(axis=-1)(state)
x = Dense(64, activation="relu")(x)
x = LayerNormalization(axis=-1)(x)
x = Dense(64, activation="relu")(x)
x = LayerNormalization(axis=-1)(x)
x = Dense(64, activation="relu")(x)
x = LayerNormalization(axis=-1)(x)
v = Dense(1, activation="linear")(x)
critic = keras.Model(inputs=state, outputs=v, name="critic")
critic.summary()

# %%
rewards = Environment()(policy)
# actor.input = Observations()(env.output)
value_loss = ValueLoss()([v])
# adv = Advantage()([v, rewards])
policy_loss = PolicyLoss()([v, act1, act2, act3])
model = keras.Model(inputs=[obs, state],
                    outputs=[policy_loss, value_loss], name="ppo")
model.summary()

# %%
plot_model(model, expand_nested=True)

# %%
with open("model.json", "w") as f:
    f.write(model.to_json())
model.save("model.h5")

# ===== UPDATE model.json ===== #

# %%
import h5py

h5f = h5py.File("model.h5", mode='r+')

with open("model.json", "r") as f:
    h5f.attrs['model_config'] = f.read()

h5f.close()

# %%
