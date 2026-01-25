训练数据在：
data/llama_hpo_data.json

即将data/llama_mpo_data.json中的"source_conversations"，"source_chosen"，"source_rejected"的value都翻译成6种目标语言，翻译脚本是src/llamafactory/data/translate_dataset.py



我翻译的bn和sw语的harmless instructions：
Multilingual-Refusal/dataset/splits_multi/harmless_train_translated_bn.json
Multilingual-Refusal/dataset/splits_multi/harmless_train_translated_sw.json
Multilingual-Refusal/dataset/splits_multi/harmless_val_translated_bn.json
Multilingual-Refusal/dataset/splits_multi/harmless_val_translated_sw.json
Multilingual-Refusal/dataset/splits_multi/harmless_test_translated_bn.json
Multilingual-Refusal/dataset/splits_multi/harmless_test_translated_sw.json

翻译脚本是：
Multilingual-Refusal/scripts/translate_data.py


2026.1.24更新：

用GPT评测safe/unsafe/invaild：把src/activation_pca_visualization_pair_lang.py脚本中的--classify_model设为"gpt"，代码最开头填好api_key，就行

每张图同时画英语和一种目标语言的脚本：
run_plot_pca.sh，不知道有没有bug了，我缺包还没跑通，但是困了要睡了

我这边examples/train_hpo/llama3.1_wpo.yaml这版效果挺好的，我写在共享文档结果里的就是这个，训出的模型在cpts_wpo-promptretain0.2-2epoch/llama3.1-8b-instruct


2026.1.25更新：
plot的结果在figs/中，运行脚本是run_plot_pca_and_distance.sh