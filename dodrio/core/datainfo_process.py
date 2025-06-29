'''
FilePath: /python-Dodrio/dodrio/core/datainfo_process.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 15:18:58
LastEditors: Yixiang Chen
LastEditTime: 2025-05-13 17:30:19
'''

import os
import pandas as pd

from dodrio.core.load_info import load_info_dict_libritts, load_info_dict_emilia, load_info_dict_tfile_supdir, load_info_dict_genshin, load_info_dict_table, load_info_dict_single_spk

BLANK_STRING='BlankNone'
info_loadfn_dict = {
    'libritts': load_info_dict_libritts,
    'emilia': load_info_dict_emilia,
    'supdir': load_info_dict_tfile_supdir,
    'genshin': load_info_dict_genshin,
    'table': load_info_dict_table,
    'singlespk': load_info_dict_single_spk,
}

################# other info save ####################

def load_pack_dict(input_dir, from_type='parquet'):
    if from_type == 'parquet':
        load_file = os.path.join(input_dir, 'utt2parquet.list')
    else:
        load_file = os.path.join(input_dir, 'uttinfo.list')
    pack_dict = {}
    with open(load_file, 'r') as lf:
        lines = lf.readlines()
    for line in lines:
        spl = line.strip().split('|')
        name= spl[0]
        packname = spl[1].split('.')[0]
        pack_dict.setdefault(packname, []).append(name)
    return pack_dict

def fake_func(info_dir, kl):
    return 0

def gen_infodir(input_dir, info_dir, out_dir, info_type, kl=['text'], lang='nolang', from_type='parquet', info_func=fake_func):
    '''
        descripttion: 
        param {*} input_dir: data dir 
        param {*} info_dir: text and info original dir
        param {*} out_dir: out_dir
        param {*} info_typ
        param {*} kl
        param {*} lang
        param {*} from_type
        return {*}
    '''
    os.makedirs(out_dir, exist_ok=True)
    pack_dict = load_pack_dict(input_dir, from_type)
    if info_type in info_loadfn_dict.keys():
        fun_load_info = info_loadfn_dict[info_type]
    else:
        fun_load_info = info_func 
    info_dict, keys_list = fun_load_info(info_dir, kl) 
    info_list_file = os.path.join(out_dir, 'uttinfo_text.list')
    opof = open(info_list_file, 'w')
    if (lang != 'nolang') and ('language' not in keys_list):
        outline = '|'.join(keys_list) + '|package_id|language\n'
    else:
        outline = '|'.join(keys_list) + '|package_id\n'
    opof.write(outline)
    for packid in pack_dict.keys():
        uttlist = pack_dict[packid]
        out_df_path = os.path.join(out_dir, packid+'.info')
        df = pd.DataFrame()
        for kk in keys_list:
            #keydata = [info_dict[utt][kk] for utt in uttlist]
            keydata = []
            for utt in uttlist:
                if utt in info_dict.keys():
                    keydata.append(info_dict[utt][kk])
                else:
                    keydata.append(None)
                    print(f'There is no {utt} info for {kk}')
            df[kk] = keydata
        if (lang != 'nolang') and ('language' not in keys_list):
            df['language'] = [lang] * len(uttlist) 
        df.to_parquet(out_df_path)
        print(f'{packid}.info had been Saved')
        for utt in uttlist:
            if utt in info_dict.keys():
                outlist = [str(info_dict[utt][kk]) if info_dict[utt][kk] else BLANK_STRING for kk in keys_list]
                outlist.append(str(packid+'.info'))
                if (lang != 'nolang') and ('language' not in keys_list):
                    outlist.append(lang)
                outline = '|'.join(outlist) + '\n'
                opof.write(outline)
    opof.close()

def load_info_dict_generalfile(use_file, kl):
    info_dict = {}
    with open(use_file, 'r') as ouf:
        lines = ouf.readlines()
    for line in lines:
        spl = line.strip().split('|')
        wavp, spk, lang, text = spl[0], spl[1], spl[2], spl[3]
        uttid = os.path.split(wavp)[-1].split('.')[0]
        useinfo = {'id':uttid, 'speaker':spk, 'text':text, 'language':lang}
        info_dict[uttid] = useinfo
    keys_list = ['id', 'speaker', 'text', 'language']
    return info_dict, keys_list
