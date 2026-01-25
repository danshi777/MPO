import json
from tqdm import tqdm
import random
from datasets import load_dataset
from deep_translator import GoogleTranslator

def load_advbench():
    data = load_dataset("walledai/AdvBench", split="train", cache_dir="data/advbench")
    return data

data = load_advbench()

target_langs =  ['zh-CN', 'ja', 'ko', 'ar', 'bn', 'sw']

translated_data = {
    "en": [],
    "zh": [],
    "ko": [],
    "ar": [],
    "bn": [],
    "sw": [],
}

for example in data:
    prompt = example["prompt"]
    for target_lang in target_langs:
        # print(f"Translateing to {target_lang}")
        translator = GoogleTranslator(source='en', target=target_lang)

# data = data[:12]
translated_data = []
skipped_cnt = 0
for i in tqdm(range(0, len(data), 6)): # 每6条是一条英文数据的对应翻译
    example = data[i]
    conversation = example["source_conversations"][0]["value"]
    chosen = example["source_chosen"]["value"]
    rejected = example["source_rejected"]["value"]
    random.shuffle(target_langs)
    for target_lang in target_langs:
        # print(f"Translateing to {target_lang}")
        translator = GoogleTranslator(source='en', target=target_lang)
        try:
            target_conversation = translator.translate(conversation)
        except TranslationNotFound as e:
            skipped_cnt += 1
            # 记录日志（可选，但强烈建议）
            print(f"[Translation skipped], skipped content: {conversation}, {e}")

            # 直接跳过该样本
            continue
        target_chosen = translator.translate(chosen)
        target_rejected = translator.translate(rejected)
        new_example = {
            "source_language": "en",
            "source_conversations": [
                {
                    "from": "human",
                    "value": conversation
                }
            ],
            "source_chosen": {
                "from": "gpt",
                "value": chosen
            },
            "source_rejected": {
                "from": "gpt",
                "value": rejected
            },
            "target_language": target_lang,
            "target_conversations": [
                {
                    "from": "human",
                    "value": target_conversation
                }
            ],
            "target_chosen": {
                "from": "gpt",
                "value": target_chosen
            },
            "target_rejected": {
                "from": "gpt",
                "value": target_rejected
            }
        }
        translated_data.append(new_example)

print(f"Total skipped samples: {skipped_cnt}")


    # 没有日语
    data = {
        "en": [],
        "zh": [],
        "ko": [],
        "ar": [],
        "bn": [],
        "sw": [],
    }

    for ex in raw_data:
        data.append(ex[lang])

    return data


with open(f"/netscratch/dshi/projects/MPO/data/llama_hpo_data.json", 'w', encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)