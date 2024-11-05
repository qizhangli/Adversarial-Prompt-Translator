
evalpath=$(python3 eval_convert_batchout.py $1)
python3 eval_harmbench.py --log_path ${evalpath}
