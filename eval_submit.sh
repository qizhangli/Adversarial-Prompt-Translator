
batch_path=$(python3 eval_get_batch.py ${logpath} ${model})
python3 eval_openai.py --submit --log_path ${batch_path}
