LOG_FOLDER='zork1_roberta'
GAME='zork1.z5'
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='lm_actor_critic'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

WEIGHTS_FILE="XTX/wandb/run-20230501_235619-12igeyka/files/weights_9.pt"
MEMORY_FILE="XTX/wandb/run-20230501_235619-12igeyka/files/memory_9.pkl"
RUN_ID="12igeyka"

python3 -m scripts.train_rl --output_dir logs/${LOG_FOLDER} \
                    --rom_path games/${GAME} \
                    --seed ${SEED} \
                    --jericho_seed ${JERICHO_SEED} \
                    --model_name ${MODEL_NAME} \
                    --eval_freq 10000000 \
                    --jericho_add_wt ${JERICHO_ADD_WT} \
                    --weight_file "$WEIGHT_FILE" \
                    --memory_file "$MEMORY_FILE" \
                    --run_id "$RUN_ID"