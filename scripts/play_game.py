# Built-in imports
import argparse
import random
import logging

# Third party imports
import jericho
import torch
import numpy as np
import wandb
import time

from pathlib import Path
import json

# Custom imports
from agents import (
    DrrnAgent,
    DrrnInvDynAgent,
    DrrnGraphInvDynAgent,
    LMActorCriticAgent,
    LMDrrnAgent,
    GraphLMActorCriticAgent,
    LMDrrnGraphInvDynAgent
)

from trainers import (
    DrrnTrainer,
    DrrnInvDynTrainer,
    DrrnGraphInvDynTrainer,
    LMActorCriticTrainer,
    GraphLMActorCriticTrainer,
    LMDrrnGraphTrainer
)

from transformers import GPT2LMHeadModel, GPT2Config

import definitions.defs as defs
from utils.env import JerichoEnv
from utils.vec_env import VecEnv
from utils import logger
from utils.memory import State, Transition
from utils.perturbations import Synset, Paraphraser, Simplifier, Shuffler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.CRITICAL)
# torch.autograd.set_detect_anomaly(True)


def configure_logger(args):
    """
    Setup various logging channels (wandb, text files, etc.).
    """
    log_dir = args.output_dir
    wandb_on = args.wandb

    type_strs = ["json", "stdout"]
    if wandb_on and log_dir != "logs":
        type_strs += ["wandb"]
    tb = logger.Logger(
        log_dir,
        [
            logger.make_output_format(type_str, log_dir, args=args)
            for type_str in type_strs
        ],
    )

    if wandb_on:
        logger.configure("{}-{}".format(log_dir, wandb.run.id),
                        format_strs=["log", "stdout"], off=args.logging_off)
    log = logger.log

    return tb, log


def load_agent_weights(agent, weights_file):
    """
    Loads weights into the agent's network.

    Args:
        agent (agents.Agent) : The initialized agent.
        weights_file (str) : Path to the weights.
    
    Returns:
        agent (agents.Agent) : The agents with loaded weights.
    """
    agent.network.load_state_dict(
        torch.load(weights_file, map_location=device),
        strict=False
    )
    return agent
    # try:
    #     api = wandb.Api()
    #     run = api.run(f"princeton-nlp/text-games/{run_id}")
    #     run.file(f"{weight_file}.pt").download(wandb.run.dir)
    #     run.file(f"{memory_file}.pkl").download(wandb.run.dir)

    #     self.memory = pickle.load(
    #         open(pjoin(wandb.run.dir, f"{memory_file}.pkl"), 'rb'))
    #     self.network.load_state_dict(
    #         torch.load(pjoin(wandb.run.dir, f"{weight_file}.pt")))
    # except Exception as e:
    #     self.log(f"Error loading model {e}")
    #     logging.error(traceback.format_exc())
    #     raise Exception("Didn't properly load model!")

def play_game(trainer, nb_episodes=10, apply=None):
    """
    Train the agent.

    Args:
        trainer (trainer) : The trainer
        nb_episodes (int) : Number of evaluation episodes
        apply (perturbations.Model) : Any perturbation model that has a method named `generate()` that receives a list of sentences and returns a list of sentences.
    
    Returns:
        eval_scores (list[float]) : Evaluation scores of `nb_episodes` episodes.
        eval_steps (list[int]) : Number of steps of each episode.
    """
    eval_scores = list()
    eval_steps = list()

    for i_ep in range(nb_episodes):
        max_score, max_eval, trainer.env_steps = 0, 0, 0
        obs, infos, states, valid_ids, transitions = trainer.setup_env(trainer.envs)

        for step in range(1, trainer.max_steps + 1):
            print(trainer.envs.get_cache_size())
            trainer.steps = step
            trainer.log("Step {}".format(step))

            # print("STATES", states)
            # print(trainer.agent.tokenizer.convert_ids_to_tokens(states[0].obs))
            # print("VALID IDS", valid_ids)
            # print("INFO['valid']", infos[0]['valid'])

            if apply:
                # Decode all strings -- they have already been encoded by this point
                obs_str = [trainer.agent.tokenizer.decode(s.obs) for s in states]
                inventory_str = [trainer.agent.tokenizer.decode(s.inventory) for s in states]
                description_str = [trainer.agent.tokenizer.decode(s.description) for s in states]
                valid_str = [[trainer.agent.tokenizer.decode(vid) for vid in valids] for valids in valid_ids]

                # Apply perturbations
                obs_p = apply.generate(obs_str)
                inventory_p = apply.generate(inventory_str)
                description_p = apply.generate(description_str)
                valid_p = [apply.generate(v_str) for v_str in valid_str]

                print(obs_p)
                print(inventory_p)
                print(description_p)
                print(valid_p)

                enc_obs = [trainer.agent.tokenizer.encode(o) for o in obs_p]
                enc_inventory = [trainer.agent.tokenizer.encode(i) for i in inventory_p]
                enc_description = [trainer.agent.tokenizer.encode(d) for d in description_p]
                enc_valid_ids = [[trainer.agent.tokenizer.encode(v) for v in vids] for vids in valid_p]

                # Build states after perturbation
                states_p = [State(enc_obs[i], enc_description[i], enc_inventory[i], infos[i]['score']) for i in range(len(states))]

                # NOTE: We must keep info['valid'] intact when passing it to act() because we need to send the original strings to Jericho
                action_ids, action_idxs, action_qvals = trainer.agent.act(states_p,
                                                                        enc_valid_ids,
                                                                        [info['valid']
                                                                            for info in infos],
                                                                        sample=True)
            else:
                action_ids, action_idxs, action_qvals = trainer.agent.act(states,
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
            trainer.log('Q-Values: {}'.format(s))

            # Update all envs
            infos, next_states, next_valids, max_score, obs = trainer.update_envs(
                action_strs, action_ids, states, max_score, transitions, obs, infos)
            states, valid_ids = next_states, next_valids

            if len(transitions[0]) == 0:  # Episode ended
                print("*** NEW EPISODE", i_ep)
                print("---> Scores", max_score)
                print("---"*10)
                eval_scores.append(max_score)
                eval_steps.append(step)
                break
    return eval_scores, eval_steps


def parse_args():
    parser = argparse.ArgumentParser()

    # Perturbation settings

    parser.add_argument("--perturbation", default=None,
                        choices=["substitute", "paraphrasing", "shuffle",
                                 "simplify"])
    parser.add_argument("--num_eval_episodes", default=1, type=int)

    # General Settings
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--rom_path', default='./games/detective.z5')
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--save_path', default='princeton-nlp/text-games/')
    parser.add_argument('--logging_off', default=0, type=int)
    parser.add_argument('--weight_file', default=None, type=str)
    parser.add_argument('--memory_file', default=None, type=str)
    parser.add_argument('--traj_file', default=None, type=str)
    parser.add_argument('--run_id', default=None, type=str)
    parser.add_argument('--project_name', default='xtx', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--jericho_add_wt', default='add_wt', type=str)

    # Environment settings
    parser.add_argument('--check_valid_actions_changed', default=0, type=int)

    # Training Settings
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--dynamic_episode_length', default=0, type=int)
    parser.add_argument('--episode_ext_type', default='steady_50', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--jericho_seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--q_update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--target_update_freq', default=100, type=int)
    parser.add_argument('--dump_traj_freq', default=5000, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--memory_alpha', default=.4, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--no_invalid_act_detect', default=0, type=int)
    parser.add_argument('--filter_invalid_acts', default=1, type=int)
    parser.add_argument('--start_from_reward', default=0, type=int)
    parser.add_argument('--start_from_wt', default=0, type=int)
    parser.add_argument('--filter_drop_acts', default=0, type=int)

    # Action Model Settings
    parser.add_argument('--max_acts', default=5, type=int)
    parser.add_argument('--tf_embedding_dim', default=128, type=int)
    parser.add_argument('--tf_hidden_dim', default=128, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--feedforward_dim', default=512, type=int)
    parser.add_argument('--tf_num_layers', default=3, type=int)
    parser.add_argument('--ngram', default=3, type=int)
    parser.add_argument('--traj_k', default=1, type=int)
    parser.add_argument('--action_model_update_freq', default=1e9, type=int)
    parser.add_argument('--smooth_alpha', default=0.00001, type=float)
    parser.add_argument('--cut_beta_at_threshold', default=0, type=int)
    parser.add_argument('--action_model_type', default='ngram', type=str)
    parser.add_argument('--tf_num_epochs', default=50, type=int)
    parser.add_argument(
        '--turn_action_model_off_after_falling', default=0, type=int)
    parser.add_argument('--traj_dropout_prob', default=0, type=float)
    parser.add_argument('--init_bin_prob', default=0.1, type=float)
    parser.add_argument('--num_bins', default=0, type=int)
    parser.add_argument('--binning_prob_update_freq', default=1e9, type=int)
    parser.add_argument('--random_action_dropout', default=0, type=int)
    parser.add_argument('--use_multi_ngram', default=0, type=int)
    parser.add_argument('--use_action_model', default=0, type=int)
    parser.add_argument('--sample_action_argmax', default=0, type=int)
    parser.add_argument('--il_max_context', default=512, type=int)
    parser.add_argument('--il_k', default=5, type=int)
    parser.add_argument('--il_batch_size', default=64, type=int)
    parser.add_argument('--il_lr', default=1e-3, type=float)
    parser.add_argument('--il_max_num_epochs', default=200, type=int)
    parser.add_argument('--il_num_eval_runs', default=3, type=int)
    parser.add_argument('--il_eval_freq', default=300, type=int)
    parser.add_argument('--il_vocab_size', default=2000, type=int)
    parser.add_argument('--il_temp', default=1., type=float)
    parser.add_argument('--use_il', default=0, type=int)
    parser.add_argument('--il_len_scale', default=1.0, type=float)
    parser.add_argument('--use_il_graph_sampler', default=0, type=int)
    parser.add_argument('--use_il_buffer_sampler', default=1, type=int)
    parser.add_argument('--il_top_p', default=0.9, type=float)
    parser.add_argument('--il_use_dropout', default=0, type=int)
    parser.add_argument('--il_use_only_dropout', default=0, type=int)

    # DRRN Model Settings
    parser.add_argument('--drrn_embedding_dim', default=128, type=int)
    parser.add_argument('--drrn_hidden_dim', default=128, type=int)
    parser.add_argument('--use_drrn_inv_look', default=1, type=int)
    parser.add_argument('--use_counts', default=0, type=int)
    parser.add_argument('--reset_counts_every_epoch', default=0, type=int)
    parser.add_argument('--sample_uniform', default=0, type=int)
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--rotating_temp', default=0, type=int)
    parser.add_argument('--augment_state_with_score', default=0, type=int)

    # Graph Model Settings
    parser.add_argument('--graph_num_explore_steps', default=50, type=int)
    parser.add_argument('--graph_rescore_freq', default=500, type=int)
    parser.add_argument('--graph_merge_freq', default=500, type=int)
    parser.add_argument('--graph_hash', default='inv_loc_ob', type=str)
    parser.add_argument('--graph_score_temp', default=1, type=float)
    parser.add_argument('--graph_q_temp', default=1, type=float)
    parser.add_argument('--graph_alpha', default=0.5, type=float)
    parser.add_argument('--log_top_blue_acts_freq', default=100, type=int)

    # Offline Q Learning settings
    parser.add_argument('--offline_q_steps', default=1000, type=int)
    parser.add_argument('--offline_q_transfer_freq', default=100, type=int)
    parser.add_argument('--offline_q_eval_runs', default=10, type=int)

    # Inv-Dyn Settings
    parser.add_argument('--type_inv', default='decode')
    parser.add_argument('--type_for', default='ce')
    parser.add_argument('--w_inv', default=0, type=float)
    parser.add_argument('--w_for', default=0, type=float)
    parser.add_argument('--w_act', default=0, type=float)
    parser.add_argument('--r_for', default=0, type=float)

    parser.add_argument('--nor', default=0, type=int, help='no game reward')
    parser.add_argument('--randr', default=0, type=int,
                        help='random game reward by objects and locations within episode')
    parser.add_argument('--perturb', default=0, type=int,
                        help='perturb state and action')

    parser.add_argument('--hash_rep', default=0, type=int,
                        help='hash for representation')
    parser.add_argument('--act_obs', default=0, type=int,
                        help='action set as state representation')
    parser.add_argument('--fix_rep', default=0, type=int,
                        help='fix representation')

    # Additional Model Settings
    parser.add_argument('--model_name', default='xtx', type=str)
    parser.add_argument('--beta', default=0.3, type=float)
    parser.add_argument('--beta_trainable', default=0, type=int)
    parser.add_argument(
        '--eps',
        default=0,
        type=int,
        help='0: ~ softmax act_value; 1: eps-greedy-exploration',
    )
    parser.add_argument(
        '--eps_type',
        default='uniform',
        type=str,
        help='uniform (-1): uniform exploration; softmax_lm (0): ~ softmax lm_value; uniform_lm_topk (>0): ~ uniform(top k w.r.t. lm_value)',
    )
    parser.add_argument(
        '--alpha',
        default=0,
        type=float,
        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now',
    )
    parser.add_argument('--sample_argmax',
                        default=0,
                        type=int,
                        help='whether to replace sampling with argmax')

    # LM params
    parser.add_argument("--lm_model_name",
                        default="sentence-transformers/all-distilroberta-v1",
                        type=str,
                        help="Name of the model to load")

    return parser.parse_args()


def main():
    assert jericho.__version__.startswith(
        "3"), "This code is designed to be run with Jericho version >= 3.0.0."

    args = parse_args()
    print(args)
    print("device", device)
    print(args.model_name)

    # Set seed across imports
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set resume (if any)
    if args.run_id:
        wandb.init(id=args.run_id, resume=True, allow_val_change=True)
        wandb.config.update({'allow_val_change': True})


    # Start logger
    tb, log = configure_logger(args)

    if args.debug:
        import pdb
        pdb.set_trace()

    # Setup envs
    cache = dict()
    eval_env = JerichoEnv(args.rom_path,
                          args.env_step_limit,
                          get_valid=True,
                          seed=args.jericho_seed,
                          args=args,
                          cache=cache,
                          start_from_reward=args.start_from_reward,
                          start_from_wt=args.start_from_wt,
                          log=log)
    envs = [
        JerichoEnv(args.rom_path,
                   args.env_step_limit,
                   get_valid=True,
                   cache=cache,
                   args=args,
                   seed=args.jericho_seed,
                   start_from_reward=args.start_from_reward,
                   start_from_wt=args.start_from_wt,
                   log=log) for _ in range(args.num_envs)
    ]

    # Setup rl model
    if args.model_name == defs.LM_AC:
        envs = VecEnv(args.num_envs, eval_env)
        agent = LMActorCriticAgent(tb, log, args, envs, None)
        trainer = LMActorCriticTrainer(tb, log, agent, envs, eval_env, args)
    
    elif args.model_name == defs.LM_DRRN:
        envs = VecEnv(args.num_envs, eval_env)
        agent = LMDrrnAgent(tb, log, args, envs, None)
        trainer = DrrnTrainer(tb, log, agent, envs, eval_env, args)
        
    elif args.model_name == defs.LM_XTX:
        assert args.use_il == args.use_action_model, "action model stuff should be on when using IL."
        assert args.r_for > 0, "r_for needs to be ON when using inverse dynamics."
        if args.il_use_dropout or args.il_use_only_dropout:
            assert args.il_use_dropout != args.il_use_only_dropout, "cannot use two types of dropout at the same time."

        envs = VecEnv(args.num_envs, eval_env)

        config = GPT2Config(vocab_size=args.il_vocab_size, n_embd=args.tf_embedding_dim,
                            n_layer=args.tf_num_layers, n_head=args.nhead, n_positions=args.il_max_context, n_ctx=args.il_max_context)
        lm = GPT2LMHeadModel(config)
        lm.train()
        agent = LMDrrnGraphInvDynAgent(args, tb, log, envs, action_models=lm, model_name=args.lm_model_name)
        trainer = LMDrrnGraphTrainer(tb, log, agent, envs, eval_env, args)
    
    elif args.model_name == defs.DRRN:
        assert args.use_action_model == 0, "'use_action_model' needs to be OFF"
        assert args.r_for == 0, "r_for needs to be zero when NOT using inverse dynamics."
        assert args.use_il == 0, "no il should be used when running DRRN."

        envs = VecEnv(args.num_envs, eval_env)

        agent = DrrnAgent(tb, log, args, envs, None)
        trainer = DrrnTrainer(tb, log, agent, envs, eval_env, args)

    elif args.model_name == defs.XTX:
        assert args.use_il == args.use_action_model, "action model stuff should be on when using IL."
        assert args.r_for > 0, "r_for needs to be ON when using inverse dynamics."
        if args.il_use_dropout or args.il_use_only_dropout:
            assert args.il_use_dropout != args.il_use_only_dropout, "cannot use two types of dropout at the same time."

        envs = VecEnv(args.num_envs, eval_env)

        config = GPT2Config(vocab_size=args.il_vocab_size, n_embd=args.tf_embedding_dim,
                            n_layer=args.tf_num_layers, n_head=args.nhead, n_positions=args.il_max_context, n_ctx=args.il_max_context)
        lm = GPT2LMHeadModel(config)
        lm.train()
        agent = DrrnGraphInvDynAgent(args, tb, log, envs, action_models=lm)
        trainer = DrrnGraphInvDynTrainer(tb, log, agent, envs, eval_env, args)
    
    elif args.model_name == defs.INV_DY:
        assert args.r_for > 0, "r_for needs to be ON when using inverse dynamics."
        assert args.use_action_model == 0, "'use_action_model' needs to be OFF."

        envs = VecEnv(args.num_envs, eval_env)

        agent = DrrnInvDynAgent(args, None, tb, log, envs)
        trainer = DrrnInvDynTrainer(tb, log, agent, envs, eval_env, args)

    else:
        raise Exception("Unknown model type!")

    # if args.weight_file is not None and args.memory_file is not None:
    #     agent.load(args.run_id, args.weight_file, args.memory_file)
    #     log("Successfully loaded network and replay buffer from checkpoint!")
    
    if args.weight_file is None:
        print("Warning: no weight file provided!")
    else:
        agent = load_agent_weights(agent, args.weight_file)

    try:
        # trainer.train()
        if args.perturbation == "shuffle":
            apply = Shuffler()
        elif args.perturbation == "substitute":
            entities = {obj.name for obj in eval_env.env.get_world_objects()}
            apply = Synset(entities=entities)
        elif args.perturbation == "paraphrasing":
            apply = Paraphraser()
        elif args.perturbation == "simplify":
            apply = Simplifier()
        else:
            apply = None
        eval_scores, eval_steps = play_game(trainer, 
                                            nb_episodes=args.num_eval_episodes, 
                                            apply=apply)
        output_data = {"scores": eval_scores, "scores_mean": np.mean(eval_scores),
                       "steps": eval_steps,
                       "steps_mean": np.mean(eval_steps)}
        outfile = Path(args.output_dir).joinpath("eval_results_%s.json" % args.perturbation)
        with open(outfile, "w") as fout:
            json.dump(output_data, fout)

    finally:
        for ps in envs.ps:
            ps.terminate()


if __name__ == "__main__":
    main()
