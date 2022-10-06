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
policy = Policy()([act1, act2, act3])
actor = keras.Model(inputs=obs, outputs=policy, name="actor")
actor.summary()

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
