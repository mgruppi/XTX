#!/bin/bash
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='lm_drrn'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

MEMORY_SIZE=10000  # Override memory_size - this is the same value used for inv_dy
wandb=1  # Set wandb on (1) or off (0)

num_eval_episodes=10

perturbations=("substitute" "simplify" "shuffle" "paraphrasing")

# run_ids=("agent_weights/qoln6itk/")

for run_id in agent_weights/*/ ;
do 
    echo "$run_id"
    GAME=$(cat "$run_id"game.txt)
    LOG_FOLDER="roberta_eval_para_$GAME"

    echo "$game"
    weight_file="$run_id"/weights_20.pt

    for pert in ${perturbations[@]}
    do
        python3 -m scripts.play_game --output_dir logs/${LOG_FOLDER} \
                        --rom_path games/${GAME} \
                        --seed ${SEED} \
                        --jericho_seed ${JERICHO_SEED} \
                        --model_name ${MODEL_NAME} \
                        --eval_freq 10000000 \
                        --jericho_add_wt ${JERICHO_ADD_WT} \
                        --memory_size ${MEMORY_SIZE} \
                        --wandb ${wandb} \
                        --weight_file ${weight_file} \
                        --num_envs 1 \
                        --max_steps 1000 \
                        --num_eval_episodes "$num_eval_episodes" \
                        --perturbation "$pert"
    done

done


