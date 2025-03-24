'''
FilePath: /base-dodrio/dodrio/core/load_info.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 15:28:08
LastEditors: Yixiang Chen
LastEditTime: 2025-03-24 17:24:00
'''

import os
import json
from tqdm import tqdm

from dodrio.core.utils import get_file_list, get_file_list_tail

################ Load info from Multi-type ######################

def load_info_dict_emilia(info_dir, kl):
    json_list = get_file_list(info_dir, '.json')
    keys_list = ["id", "wav", "text", "duration", "speaker", "language", "dnsmos"]
    info_dict = {}
    for jsonf in tqdm(json_list, desc='LoadJson'):
        with open(jsonf, 'r') as ojf:
            data = json.load(ojf)
        info_dict[data["id"]] = data
    return info_dict, keys_list

def load_info_dict_tfile_supdir(info_dir, kl):
    sufdict = {'text': '_text.txt', 'punc_text': '_text_punc.txt'}
    titleid_list = os.listdir(info_dir)
    if 'data.lst' in titleid_list: 
        titleid_list.remove('data.lst')
    info_dict = {}
    punc_flag = ('punc_text' in kl)
    for title in tqdm(titleid_list, desc='LoadInfo4Supdir'):
        title_dir = os.path.join(info_dir, title)
        spk_list = os.listdir(title_dir)
        for spk in spk_list:
            spk_dir = os.path.join(title_dir, spk)
            tf_list = get_file_list_tail(spk_dir, sufdict['text'])
            for tff in tf_list:
                #basename = os.path.split(tff)[-1].split('.')[0]
                uttid = os.path.split(tff)[-1].split(sufdict['text'])[0]
                with open(tff, 'r') as otff:
                    text = otff.read().strip()
                if not punc_flag:
                    useinfo = {'id':uttid, 'titleid':title, 'speaker':spk, 'text':text}
                else:
                    ptf = os.path.join(spk_dir, uttid + sufdict['punc_text'])
                    if os.path.isfile(ptf):
                        with open(ptf, 'r') as optf: 
                            punc_text = optf.read().strip()
                    else:
                        punc_text = None
                    useinfo = {'id':uttid, 'titleid':title, 'speaker':spk, 'text':text, 'punc_text':punc_text}
                info_dict[uttid] = useinfo
    if punc_flag:
        keys_list = ['id', 'titleid', 'speaker', 'text', 'punc_text']
    else:
        keys_list = ['id', 'titleid', 'speaker', 'text']
    return info_dict, keys_list
            
def load_info_dict_genshin(info_dir, kl=['text']):
    sufdict = {'text': '.lab'}
    info_dict = {}
    spk_list = os.listdir(info_dir)
    for spk in spk_list:
        spk_dir = os.path.join(info_dir, spk)
        tf_list = get_file_list(spk_dir, sufdict['text'])
        for tff in tf_list:
            #basename = os.path.split(tff)[-1].split('.')[0]
            uttid = os.path.split(tff)[-1].split(sufdict['text'])[0]
            with open(tff, 'r') as otff:
                text = otff.read().strip()
            useinfo = {'id':uttid, 'speaker':spk, 'text':text, 'language':'ZH'}
            info_dict[uttid] = useinfo
    keys_list = ['id', 'speaker', 'text', 'language']
    return info_dict, keys_list

def load_info_dict_table(info_dir, kl):
    use_file = 'info_align.list'
    titleid_list = os.listdir(info_dir)
    if 'data.lst' in titleid_list: 
        titleid_list.remove('data.lst')
    info_dict = {}
    for title in tqdm(titleid_list, desc='LoadInfo4Supdir'):
        titleid = title.split('_')[0]
        title_dir = os.path.join(info_dir, title)
        usefile_path = os.path.join(title_dir, use_file)
        with open(usefile_path, 'r') as ouf:
            lines = ouf.readlines()
        for line in lines:
            spl = line.strip().split('|')
            wavp, spk, lang, text, phoneme, dur = spl[0], spl[1], spl[2], spl[3], spl[4], spl[5]
            uttid = os.path.split(wavp)[-1].split('.')[0]
            useinfo = {'id':uttid, 'titleid':titleid, 'speaker':spk, 'text':text, 'language':lang}
            info_dict[uttid] = useinfo
    keys_list = ['id', 'titleid', 'speaker', 'text', 'language']
    return info_dict, keys_list

def load_info_dict_libritts(info_dir, kl):
    sufdict = {'text': '.normalized.txt', 'unnorm_text': '.original.txt'}
    spkid_list = os.listdir(info_dir)
    info_dict = {}
    unnorm_flag = ('unnorm_text' in kl)
    for spk in tqdm(spkid_list, desc='LoadInfo4Supdir'):
        spk_dir = os.path.join(info_dir, spk)
        chapter_list = os.listdir(spk_dir)
        for chapter in chapter_list:
            chapter_dir = os.path.join(spk_dir, chapter)
            tf_list = get_file_list_tail(chapter_dir, sufdict['text'])
            for tff in tf_list:
                #basename = os.path.split(tff)[-1].split('.')[0]
                uttid = os.path.split(tff)[-1].split(sufdict['text'])[0]
                with open(tff, 'r') as otff:
                    text = otff.read().strip()
                if not unnorm_flag:
                    useinfo = {'id':uttid, 'chapterid':chapter, 'speaker':spk, 'text':text}
                else:
                    ptf = os.path.join(chapter_dir, uttid + sufdict['unnorm_text'])
                    if os.path.isfile(ptf):
                        with open(ptf, 'r') as optf: 
                            unnorm_text = optf.read().strip()
                    else:
                        unnorm_text = None
                    useinfo = {'id':uttid, 'chapterid':chapter, 'speaker':spk, 'text':text, 'unnorm_text':unnorm_text}
                info_dict[uttid] = useinfo
    if unnorm_flag:
        keys_list = ['id', 'chapterid', 'speaker', 'text', 'unnorm_text']
    else:
        keys_list = ['id', 'chapterid', 'speaker', 'text']
    return info_dict, keys_list