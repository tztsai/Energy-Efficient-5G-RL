import torch
import wandb
import imageio
import numpy as np
from .runner import Runner, _t2n
from utils import sys, time, trange, notice, pd


class MultiCellNetRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        pbar = trange(episodes)

        for episode in pbar:
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in trange(self.episode_length, file=sys.stdout, postfix='collecting'):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, cent_obs, reward, done, infos, avail_acts = self.envs.step(actions)

                data = obs, cent_obs, reward, done, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                if episode % 10 == 0:
                    self.save(suffix='_eps%s' % episode)

            # log information
            if episode % self.log_interval == 0:
                env_info = pd.DataFrame(list(infos)).mean()
                train_infos.update(env_info)
                step_rew = np.mean(self.buffer.rewards)
                train_infos.update(
                    average_step_reward = step_rew,
                    average_episode_reward = step_rew * self.episode_length
                )
                pbar.set_postfix(step_reward=step_rew)
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, cent_obs, avail_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            cent_obs = obs
        self.buffer.cent_obs[0] = cent_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.cent_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]))

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, cent_obs, reward, done, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

        if np.all(done):
            rnn_states[:] = 0
            rnn_states_critic[:] = 0
            masks[:] = 0

        if not self.use_centralized_V:
            cent_obs = obs

        self.buffer.insert(cent_obs, obs, rnn_states, rnn_states_critic, actions,
                           action_log_probs, values, reward, masks)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
                
    @torch.no_grad()
    def take_actions(self, obs, reset_rnn=False, _rnn_cache={}):
        n_threads = obs.shape[0]
        if reset_rnn or 'states' not in _rnn_cache:
            n_agents, rnn_layers, rnn_dim = self.buffer.rnn_states.shape[2:]
            states_shape = (n_threads * n_agents, rnn_layers, rnn_dim)
            _rnn_cache['states'] = np.zeros(states_shape, dtype=np.float32)
        rnn_states = _rnn_cache['states']
        obs = np.concatenate(obs)
        actions = self.trainer.policy.act(obs, rnn_states, deterministic=True)
        rnn_states = self.trainer.policy.get_actor_rnn_state()
        actions = np.array(np.split(actions, n_threads))
        _rnn_cache['states'] = rnn_states
        return actions

    @torch.no_grad()
    def eval(self, total_num_steps=None):
        episode_rewards = []
        
        episode_rewards = []
        one_episode_rewards = []

        for episode in range(self.all_args.eval_episodes):
            done = None
            obs, _, _ = self.eval_envs.reset()
            
            while True:
                actions = self.take_actions(obs, reset_rnn=done is None)

                obs, _, reward, done, _, _ = self.eval_envs.step(actions)
                one_episode_rewards.append(reward)

                if done.all(): break
                
            episode_rewards.append(np.sum(one_episode_rewards))
            one_episode_rewards.clear()

        episode_rewards = np.array(episode_rewards)
        eval_env_infos = {'episode_rewards': episode_rewards}
        self.log_env(eval_env_infos, total_num_steps)
        print("eval average episode reward:", episode_rewards.mean())

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for _ in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()
                actions = self.take_actions(obs)

                # Obser reward and next obs
                obs, rewards, dones, infos, _ = envs.step(actions)
                episode_rewards.append(rewards)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is:", 
                  np.mean(np.sum(np.array(episode_rewards), axis=0)))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames,
                            duration=self.all_args.ifi)
