# Built-in Imports
import time
import heapq as pq
import statistics as stats
import random
import copy
from typing import Callable, List, Dict, Union

# Libraries
import wandb
import torch
from jericho.util import clean

# Custom imports
from trainers import Trainer

from utils.util import process_action, check_exists, load_object, save_object
from utils.env import JerichoEnv
from utils.vec_env import VecEnv
from utils.memory import Transition
import utils.logger as logger
import utils.ngram as Ngram


OBJECTS_DIR = './saved_objects'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrrnTrainer(Trainer):
    def __init__(
        self,
        tb: logger.Logger,
        log: Callable[..., None],
        agent,
        envs: VecEnv,
        eval_env: JerichoEnv,
        args: Dict[str, Union[str, int, float]]
    ):
        super().__init__(tb, log, agent, envs, eval_env, args)

        # Action model settings
        self.use_action_model = args.use_action_model
        self.rotating_temp = args.rotating_temp
        if self.use_action_model:
            Ngram.init_trainer(self, args)

        self.collected_trajs = []
        self.full_traj_folder = args.output_dir.split('/')[-1][:-3]
        self.dump_traj_freq = args.dump_traj_freq

    def setup_env(self, envs):
        """
        Setup the environment.
        """
        obs, infos = envs.reset()
        if self.use_action_model:
            states = self.agent.build_states(
                obs, infos, ['reset'] * 8, [[]] * 8)
        else:
            states = self.agent.build_states(obs, infos)
        valid_ids = [self.agent.encode(info['valid']) for info in infos]
        transitions = [[] for info in infos]

        return obs, infos, states, valid_ids, transitions

    def update_envs(self, action_strs, action_ids, states, max_score: int,
                    transitions, obs, infos):
        """
        TODO
        """
        next_obs, next_rewards, next_dones, next_infos = self.envs.step(
            action_strs)

        # Remove actions from `next_infos` to prevent a bug in Zork 1
        # [[cmd for cmd in ad_cmd if 'put sack in' not in cmd and cmd != 'put all in nest'] \
        #                              for ad_cmd in admissible_commands]
        next_infos = list(next_infos)
        # Sanitize next valid moves
        for next_info in next_infos:
            next_info['valid'] = [cmd for cmd in next_info['valid'] if 'put sack in' not in cmd and cmd != 'put all in nest']

        if self.use_action_model:
            next_states = self.agent.build_states(
                next_obs, next_infos, action_strs, [state.acts for state in states])
        else:
            next_states = self.agent.build_states(next_obs, next_infos)

        next_valids = [self.agent.encode(next_info['valid'])
                       for next_info in next_infos]

        self.envs.add_full_traj(
            [
                (ob, info['look'], info['inv'], act, r) for ob, info, act, r in zip(obs, infos, action_strs, next_rewards)
            ]
        )

        if self.use_action_model:
            # Add to environment trajectory
            trajs = self.envs.add_traj(
                list(map(lambda x: process_action(x), action_strs)))

            for next_reward, next_done, next_info, traj in zip(next_rewards, next_dones, next_infos, trajs):
                # Push to trajectory memory if reward was positive and the episode didn't end yet
                if next_reward > 0:
                    Ngram.push_to_traj_mem(self, next_info, traj)

        for i, (next_ob, next_reward, next_done, next_info, state, next_state, next_action_str) in enumerate(zip(next_obs, next_rewards, next_dones, next_infos, states, next_states, action_strs)):
            # Log
            self.log('Action_{}: {}'.format(
                self.steps, next_action_str), condition=(i == 0))
            self.log("Reward{}: {}, Score {}, Done {}".format(
                self.steps, next_reward, next_info['score'], next_done), condition=(i == 0))
            self.log('Obs{}: {} Inv: {} Desc: {}'.format(
                self.steps, clean(next_ob), clean(next_info['inv']),
                clean(next_info['look'])), condition=(i == 0))

            transition = Transition(
                state, action_ids[i], next_reward, next_state, next_valids[i], next_done)
            transitions[i].append(transition)
            self.agent.observe(transition)

            if next_done:
                self.tb.logkv_mean('EpisodeScore', next_info['score'])
                if next_info['score'] >= max_score:  # put in alpha queue
                    if next_info['score'] > max_score:
                        self.agent.memory.clear_alpha()
                        max_score = next_info['score']
                    for transition in transitions[i]:
                        self.agent.observe(transition, is_prior=True)
                transitions[i] = []

                if self.use_action_model:
                    Ngram.log_recovery_metrics(self, i)

                    if self.envs.get_ngram_needs_update(i):
                        Ngram.update_ngram(self, i)

                if self.rotating_temp:
                    self.agent.network.T[i] = random.choice([1.0, 2.0, 3.0])

                next_infos = list(next_infos)
                # add finished to trajectory to collection
                traj = self.envs.add_full_traj_i(
                    i, (next_obs[i], next_infos[i]['look'], next_infos[i]['inv']))
                self.collected_trajs.append(traj)

                next_obs[i], next_infos[i] = self.envs.reset_one(i)

                if self.use_action_model:
                    next_states[i] = self.agent.build_skip_state(
                        next_obs[i], next_infos[i], 'reset', [])
                else:
                    next_states[i] = self.agent.build_state(
                        next_obs[i], next_infos[i])

                next_valids[i] = self.agent.encode(next_infos[i]['valid'])

        return next_infos, next_states, next_valids, max_score, next_obs

    def _wrap_up_episode(self, info, env, max_score, transitions, i):
        """
        Perform final logging, updating, and building for next episode.
        """
        # Logging & update
        self.tb.logkv_mean('EpisodeScore', info['score'])
        if env.max_score >= max_score:
            for t in transitions[i]:
                self.agent.observe(t, is_prior=True)
        transitions[i] = []
        self.env_steps += info["moves"]

        # Build ingredients for next step
        next_ob, next_info = env.reset()
        if self.use_action_model:
            next_state = self.agent.build_skip_state(
                next_ob, next_info, [], 'reset')
        else:
            next_state = self.agent.build_state(next_ob, next_info)
        next_valid = self.agent.encode(next_info['valid'])

        return next_state, next_valid, next_info

    def train(self):
        """
        Train the agent.
        """
        start = time.time()
        max_score, max_eval, self.env_steps = 0, 0, 0
        obs, infos, states, valid_ids, transitions = self.setup_env(self.envs)

        for step in range(1, self.max_steps + 1):
            print(self.envs.get_cache_size())
            self.steps = step
            self.log("Step {}".format(step))
            action_ids, action_idxs, action_qvals = self.agent.act(states,
                                                                   valid_ids,
                                                                   [info['valid']
                                                                       for info in infos],
                                                                   sample=True)

            # Get the actual next action string for each env
            action_strs = [
                info['valid'][idx] for info, idx in zip(infos, action_idxs)
            ]

            # Log envs[0]
            s = ''
            for idx, (act, val) in enumerate(
                    sorted(zip(infos[0]['valid'], action_qvals[0]),
                           key=lambda x: x[1],
                           reverse=True), 1):
                s += "{}){:.2f} {} ".format(idx, val.item(), act)
            self.log('Q-Values: {}'.format(s))

            # Update all envs
            infos, next_states, next_valids, max_score, obs = self.update_envs(
                action_strs, action_ids, states, max_score, transitions, obs, infos)
            states, valid_ids = next_states, next_valids

            self.end_step(step, start, max_score, action_qvals, max_eval)

    def end_step(self, step: int, start, max_score: int, action_qvals,
                 max_eval: int):
        """
        TODO
        """
        if step % self.q_update_freq == 0:
            self.update_agent()

        if step % self.target_update_freq == 0:
            self.agent.transfer_weights()

        if step % self.log_freq == 0:
            # rank_metrics = self.evaluate_optimal()
            rank_metrics = dict()
            self.write_to_logs(step, start, self.envs, max_score, action_qvals,
                               rank_metrics)

        # Save model weights etc.
        if step % self.checkpoint_freq == 0:
            self.agent.save(int(step / self.checkpoint_freq),
                            self.top_k_traj if self.use_action_model else None)

        # Evaluate agent across several runs
        if step % self.eval_freq == 0:
            eval_score = self.evaluate(nb_episodes=10)
            wandb.log({
                'EvalScore': eval_score,
                'Step': step,
                "Env Steps": self.env_steps
            })
            if eval_score >= max_eval:
                max_eval = eval_score
                self.agent.save(step, is_best=True)

        if self.use_action_model:
            Ngram.end_step(self, step)

    def write_to_logs(self, step, start, envs, max_score, qvals, rank_metrics,
                      *args):
        """
        Log any relevant metrics. 
        """
        self.tb.logkv('Step', step)
        self.tb.logkv('Env Steps', self.env_steps)
        # self.tb.logkv('Beta', self.agent.network.beta)
        for key, val in rank_metrics.items():
            self.tb.logkv(key, val)
        self.tb.logkv("FPS", int((step * len(envs)) / (time.time() - start)))
        self.tb.logkv("EpisodeScores100", self.envs.get_end_scores().mean())
        self.tb.logkv('MaxScore', max_score)
        self.tb.logkv('#UniqueActs', self.envs.get_unique_acts())
        self.tb.logkv('#CacheEntries', self.envs.get_cache_size())

        if self.use_action_model:
            Ngram.log_metrics(self)

        self.tb.dumpkvs()
