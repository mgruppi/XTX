LOG_FOLDER='zork1_roberta'
GAME='zork1.z5'
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='lm_actor_critic'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

MEMORY_SIZE=10000  # Override memory_size - this is the same value used for inv_dy

WEIGHT_FILE="wandb/run-20230504_114229-17gkgbbe/files/weights_9.pt"
MEMORY_FILE="wandb/run-20230504_114229-17gkgbbe/files/memory_9.pkl"

# # NO NEED TO PASS ABSOLUTE VALUES TO FILES
# WEIGHT_FILE="weights_9.pt"
# MEMORY_FILE="memory_9.pkl

RUN_ID="17gkgbbe"

python3 -m scripts.train_rl --output_dir logs/${LOG_FOLDER} \
                    --rom_path games/${GAME} \
                    --seed ${SEED} \
                    --jericho_seed ${JERICHO_SEED} \
                    --model_name ${MODEL_NAME} \
                    --eval_freq 10000000 \
                    --jericho_add_wt ${JERICHO_ADD_WT} \
                    --memory_size ${MEMORY_SIZE} \
                    --weight_file "$WEIGHT_FILE" \
                    --memory_file "$MEMORY_FILE" \
                    --run_id "$RUN_ID"