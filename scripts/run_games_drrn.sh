# LOG_FOLDER='zork1_roberta'
# GAME='zork1.z5'
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='drrn'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

MEMORY_SIZE=10000  # Override memory_size - this is the same value used for inv_dy

declare -a game_names=("zork3.z5" "inhumane.z5" "ludicorp.z5" "pentari.z5" "detective.z5" "balances.z5" "library.z5" "deephome.z5" "enchanter.z3" "omniquest.z5" "zork1.z5")

for game_name in "${game_names[@]}"
do
    LOG_FOLDER="drrn_$game_name"

    python3 -m scripts.train_rl --output_dir logs/${LOG_FOLDER} \
                        --rom_path games/${game_name} \
                        --seed ${SEED} \
                        --jericho_seed ${JERICHO_SEED} \
                        --model_name ${MODEL_NAME} \
                        --eval_freq 10000000 \
                        --jericho_add_wt ${JERICHO_ADD_WT} \
                        --memory_size ${MEMORY_SIZE}
done