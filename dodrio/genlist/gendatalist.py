'''
FilePath: /python-Dodrio/dodrio/genlist/gendatalist.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-02-08 17:25:21
LastEditors: Yixiang Chen
LastEditTime: 2025-03-25 15:19:07
'''

import os
import json

LANGUAGE_NONE = 'LangNone'

def checkutt(utt, pack_id, start, end, spk, text, language, ptype='pack'):
    threshold_audio = 10 # 48000 sample rate 
    if ptype == 'pack':
        if (int(end) - int(start)) > threshold_audio * 2:
            return True
    else:
        return False

def SetProcess_pack(uttinfo_text_file, package_info_file, package_dir):
    with open(uttinfo_text_file, 'r') as otif:
        ti_lines = otif.readlines()
    info_dict = {}
    index_list = ti_lines[0].strip().split('|')
    spkid, textid, langid = -1, -1, -1 
    for idx in range(len(index_list)):
        if index_list[idx] == 'speaker':
            spkid = idx
        elif index_list[idx] == 'text':
            textid = idx
        if 'language' in index_list:
            if index_list[idx] == 'language': 
                langid = idx
    for idx in range(1, len(ti_lines)):
        spl = ti_lines[idx].strip().split('|')
        utt = spl[0]
        spk = spl[spkid]
        text = spl[textid]
        language = spl[langid] if langid != -1 else LANGUAGE_NONE
        info_dict[utt] = [spk, text, language]
    del ti_lines
    with open(package_info_file, 'r') as opif:
        p_lines = opif.readlines()
    outinfo_list = []
    for idx in range(len(p_lines)):
        spl = p_lines[idx].strip().split('|')
        utt, pack_id, start, end = spl[0], spl[1], spl[2], spl[3]
        # utt_id|save_type|pack_path|start_pointer|end_pointer|spk|language|text
        spk, text, language = info_dict[utt]
        if checkutt(utt, pack_id, start, end, spk, text, language, ptype='pack'):
            outline = [utt, 'pack', os.path.join(package_dir, pack_id), start, end, spk, language, text]
            outinfo_list.append(outline)
    return outinfo_list

def SetProcess_dir(package_dir, info_dir):
    uttinfo_text_file = os.path.join(info_dir, 'uttinfo_text.list')
    package_info_file = os.path.join(package_dir, 'uttinfo.list')
    use_info_list = SetProcess_pack(uttinfo_text_file, package_info_file, package_dir)
    return use_info_list

def genListDir(supdir_list, outdir, prefix='test', subnum=50000):

    os.makedirs(outdir, exist_ok=True)
    all_list_file = os.path.join(outdir, 'all_usage_utt.list')
    sub_table_dir = os.path.join(outdir, 'subtable')
    os.makedirs(sub_table_dir, exist_ok=True)
    spk_id_file = os.path.join(outdir, 'spk_name.dict') 
    list_list_file = os.path.join(outdir, 'list_subtable.list')

    all_info_list = []
    for supdir in supdir_list:
        package_dir = os.path.join(supdir, 'pack_dir') 
        info_dir = os.path.join(supdir, 'info_dir')
        use_info_list = SetProcess_dir(package_dir, info_dir)
        all_info_list.extend(use_info_list)
    
    spkdict = {}
    spkid = 0
    save_all = True
    olistsub = open(list_list_file, 'w')
    if save_all:
        oallt = open(all_list_file, 'w')
    for idx in range(0, len(all_info_list), subnum):
        use_id = (idx+1) // subnum 
        subtable_file = os.path.join(sub_table_dir, prefix+'_sub_'+str(use_id).rjust(6,'0')+ '.table') 
        olistsub.write(subtable_file+'\n')
        osubt = open(subtable_file, 'w')
        for ik in range(idx, min(len(all_info_list), idx+subnum)):
            info_line = all_info_list[ik]
            spk = info_line[5]
            if spk not in spkdict.keys():
                spkdict[spk] = spkid
                spkid += 1
            outline = '|'.join(info_line) + '\n'
            osubt.write(outline)
            if save_all:
                oallt.write(outline)
        osubt.close()
    if save_all:
        oallt.close()
    olistsub.close()
    with open(spk_id_file, 'w', encoding="utf-8") as opf:
        json.dump(spkdict, opf, indent=2, ensure_ascii=False)
    
