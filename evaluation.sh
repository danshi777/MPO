# python src/evaluate_safety.py
# python src/evaluate_safety.py --model_id_or_path=cpts_hpo/llama3.1-8b-instruct --model_name=llama31-8b-hpo
# python src/evaluate_safety.py --model_id_or_path=cpts_wpo/llama3.1-8b-instruct --model_name=llama31-8b-wpo
# python src/evaluate_safety.py --model_id_or_path=/netscratch/dshi/projects/MultiPO/ORIMPO/cpts_mpo/llama3.1-8b-instruct --model_name=llama31-8b-mpo
# python src/evaluate_safety.py --model_id_or_path=cpts_wpo/llama3.1-8b-instruct_beta1 --model_name=llama31-8b-wpo_beta1

# python src/evaluate_safety.py --model_id_or_path=/netscratch/dshi/models/Qwen2.5-7B-Instruct --model_name=qwen25-7b
# python src/evaluate_safety.py --model_id_or_path=cpts_mpo/qwen2.5-7b-instruct --model_name=qwen25-7b-mpo

# python src/evaluate_safety.py --model_id_or_path=cpts_simpo/llama3.1-8b-instruct --model_name=llama31-8b-simpo

# python src/evaluate_safety.py --model_id_or_path=cpts_wpo-noretain/llama3.1-8b-instruct --model_name=llama31-8b-wpo_noretain

python src/evaluate_safety.py --model_id_or_path=cpts_wpo-promptretain0.2-2epoch/llama3.1-8b-instruct --model_name=llama31-8b-wpo-promptretain0.2-2epoch