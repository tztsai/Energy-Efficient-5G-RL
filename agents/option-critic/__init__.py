from .agent.OptionCritic_agent import OptionCriticAgent
from .component import *
from .network import *
from .utils import *


def make_option_critic_agent(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('game', 'DefaultGame')
    kwargs.setdefault('log_level', 0)
    config = Config().merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    return OptionCriticAgent(config)
