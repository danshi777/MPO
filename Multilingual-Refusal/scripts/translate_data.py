from dotenv import load_dotenv
load_dotenv(override=True)
import os
import sys
# os.chdir(os.path.dirname('/mounts/Users/student/xinpeng/gdata/code/refusal-multilingual/'))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


from dataset.load_dataset import load_dataset_split
import deepl
import json
from tqdm import tqdm
from deep_translator import GoogleTranslator



# translate the harmful_test set to the target language
def translate_harmful_to_target_language(target_lang, split):
    translator = GoogleTranslator(source='en', target=target_lang)
    if data_type == 'harmless':
        if split == 'test':
            split_ = split + '_500_sampled'
        else :
            split_ = split + '_200_sampled'
    else:
        split_ = split
    harmful_test_set = load_dataset_split(data_type, split=split_)
    harmful_test_set_translated = []
    
    if split == 'test':
    
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translator.translate(item['instruction'])
                to_add = {
                    'instruction': item['instruction'],
                    'instruction_translated': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except Exception as e:
                print(f"Could not translate: {item['instruction']}")
                print(f"Error: {e}")
                continue
    else:
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translator.translate(item['instruction'])
                to_add = {
                    'instruction': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except:
                continue
        # print(f"Translating: {item['instruction']} -> {translated_item.text}")
        # break
    return harmful_test_set_translated

# 'en', 'de','es', 'fr','it', 'nl', 'pl',  'ar','th',  'yo', 'ru', 
# target_langs =  [ 'zh-CN', 'ja','ko',]
target_langs = ['bn', 'sw']
#  'th',  'yo', 'ru', 'zh', 'ja','ko',
# ['de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
# target_langs = []



for target_lang in target_langs:
    for data_type in [ "harmless"]:
        for split in ['train', 'test', 'val']:
            print(f"Translating {data_type}_{split} to {target_lang}")
            harmful_test_set_translated = translate_harmful_to_target_language(target_lang, split)
            with open(f'dataset/splits_multi/{data_type}_{split}_translated_{target_lang}.json', 'w') as f:
                json.dump(harmful_test_set_translated, f, indent=4)


# GoogleTranslator().get_supported_languages(as_dict=True):
# {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}