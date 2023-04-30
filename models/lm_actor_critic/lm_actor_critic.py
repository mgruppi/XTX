# Built-in Imports
import itertools
from typing import Callable, Dict, Union, List

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Imports
from utils import logger
import utils.ngram as Ngram
import utils.lm_actor_critic as AC
from utils.memory import State, StateWithActs
from utils.env import JerichoEnv

import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LMActorCriticQNetwork(nn.Module):
    def __init__(self,
                 tb: logger.Logger,
                 log: Callable[..., None],
                 vocab_size: int,
                 envs: List[JerichoEnv],
                 action_models,
                 tokenizer,
                 text_encoder,
                 args: Dict[str, Union[str, int, float]]):
        super(LMActorCriticQNetwork, self).__init__()
        self.sample_argmax = args.sample_argmax
        self.sample_uniform = args.sample_uniform
        self.envs = envs

        self.log = log
        self.tb = tb

        AC.init_model(self, args, vocab_size, tokenizer, text_encoder)

        self.use_action_model = args.use_action_model
        if self.use_action_model:
            Ngram.init_model(self, action_models, args)

    def forward(self, state_batch, act_batch):

        time_total = time.time()

        # Zip the state_batch into an easy access format
        if self.use_action_model:
            state = StateWithActs(*zip(*state_batch))
        else:
            state = State(*zip(*state_batch))

        time_enc_act = time.time()
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = AC.packed_rnn(self, act_batch, self.act_encoder)
        time_enc_act = time.time() - time_enc_act
        # act_out is a n_actions X embedding dim tensor

        # Encode the various aspects of the state

        # Obs, description inv and act are already tokenized
        # state.obs is an n_batches x L size tuple of lists
        # obs_out is an n_batches X embedding_dim tensor
        time_enc_obs = time.time()
        obs_out = AC.packed_rnn(self, state.obs, self.obs_encoder)
        time_enc_obs = time.time() - time_enc_obs
        time_enc_look = time.time()
        look_out = AC.packed_rnn(self, state.description, self.look_encoder)
        time_enc_look = time.time() - time_enc_look
        time_enc_inv = time.time()
        inv_out = AC.packed_rnn(self, state.inventory, self.inv_encoder)
        time_enc_inv = time.time() - time_enc_inv

        state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        # Expand the state to match the batches of actions
        state_out = torch.cat(
            [state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)],
            dim=0)

        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        ac_scores = self.act_scorer(z).squeeze(-1)

        # print("- Forward times")
        # print("  + Enc acts: % .3f" % time_enc_act)
        # print("  + Enc obs: % .3f" % time_enc_obs)
        # print("  + Enc look: % .3f" % time_enc_look)
        # print("  + Enc inv: % .3f" % time_enc_inv)
        time_total = time.time() - time_total
        print(" - forward() time  %.3f" % time_total )

        # Split up the q-values by batch
        return ac_scores.split(act_sizes)

    @torch.no_grad()
    def act(
        self,
        states: List[Union[State, StateWithActs]],
        valid_ids: List[List[List[int]]],
        valid_strs: List[List[str]],
        graph_masks=None
    ):
        """
        Returns an action-string, optionally sampling from the distribution
        of Q-Values.
        """
        return AC.act(self, states, valid_ids, valid_strs, self.log, graph_masks)
