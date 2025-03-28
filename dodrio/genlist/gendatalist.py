'''
FilePath: /python-Dodrio/dodrio/genlist/gendatalist.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-02-08 17:25:21
LastEditors: Yixiang Chen
LastEditTime: 2025-03-28 10:10:02
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
    
########################  new version  ###################################

def pack_dict_load(package_dir):
    package_info_file = os.path.join(package_dir, 'uttinfo.list')
    with open(package_info_file, 'r') as opif:
        p_lines = opif.readlines()
    outinfo_dict = {}
    for idx in range(len(p_lines)):
        spl = p_lines[idx].strip().split('|')
        utt, pack_id, start, end = spl[0], spl[1], spl[2], spl[3]
        packpath = os.path.join(package_dir, pack_id)
        outinfo_dict[utt] = [packpath, start, end]
    keylist = ['packpath', 'start', 'end']
    return outinfo_dict, keylist 

def info_dict_load(info_dir):
    uttinfo_text_file = os.path.join(info_dir, 'uttinfo_text.list')
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
    keylist = ['speaker', 'text', 'language']
    del ti_lines 
    return info_dict, keylist

def feat_dict_load(feat_dir, featname):
    featinfo_file = os.path.join(feat_dir, 'feat_info_'+ featname +'.list')
    with open(featinfo_file, 'r') as ofif:
        info_lines = ofif.readlines()
    info_dict = {}
    for line in info_lines:
        spl = line.strip().split('|')
        utt, packid, start, end, shape = spl[0], spl[1], spl[2], spl[3], spl[4]
        packpath = os.path.join(feat_dir, packid)
        info_dict[utt] = [packpath, start, end, shape]
    keylist = ['packpath', 'start', 'end', 'shape' ]
    return info_dict, keylist 

def datadirProcess(datadir, featlist, check_func):
    all_keylist = []
    package_dir = os.path.join(datadir, 'pack_dir')
    wav_info_dict, kl1 = pack_dict_load(package_dir)
    all_keylist.extend(kl1)
    info_dir = os.path.join(datadir, 'info_dir')
    info_dict, kl2 = info_dict_load(info_dir)
    all_keylist.extend(kl2)
    supfeat_dict = {}
    kl_dict = {}
    for featname in featlist:
        feat_dir = os.path.join(datadir, featname+'_dir')
        supfeat_dict[featname], kl_dict[featname] =  feat_dict_load(feat_dir, featname)
        all_keylist.append('featname_'+featname)
        all_keylist.extend(kl_dict[featname])
    
    out_dict = {}
    intersection = wav_info_dict.keys() & info_dict.keys() 
    for featname in featlist:
        intersection = intersection & supfeat_dict[featname].keys() 
    
    for utt in intersection:
        outinfo_list = wav_info_dict[utt]
        outinfo_list.extend(info_dict[utt])
        for featname in featlist:
            outinfo_list.append(featname)
            outinfo_list.extend(supfeat_dict[featname][utt])
        if check_func(outinfo_list):
            out_dict[utt] = outinfo_list
    return out_dict, all_keylist


def check_func(outinfo_list):
    return True

def gen_datalist(supdir_list, outdir, featlist, check_func, prefix, subnum=50000):
    os.makedirs(outdir, exist_ok=True)
    allsave_flag = True
    if allsave_flag:
        all_list_file = os.path.join(outdir, 'all_usage_utt.list')
    sub_table_dir = os.path.join(outdir, 'subtable')
    os.makedirs(sub_table_dir, exist_ok=True)
    spk_id_file = os.path.join(outdir, 'spk_name.dict') 
    list_list_file = os.path.join(outdir, 'list_subtable.list')
    keys_file = os.path.join(outdir, 'keys_name') 

    spkdict = {}
    spkid = 0
    subtabel_id = 0
    tmp_idx = 0
    olistsub = open(list_list_file, 'w')
    if allsave_flag:
        oallt = open(all_list_file, 'w')
    for datadir in supdir_list:
        out_dict, all_keylist = datadirProcess(datadir, featlist, check_func)
        for utt in out_dict.keys():
            if tmp_idx >=subnum:
                tmp_idx = 0
                opsubt.close()
            if tmp_idx == 0:
                subtable_file = os.path.join(sub_table_dir, prefix+'_sub_'+str(subtabel_id).rjust(6,'0')+ '.table')  
                opsubt = open(subtable_file, 'w')
                olistsub.write(subtable_file+'\n')
            
            outlist = [utt]
            outlist.extend(out_dict[utt])
            spk = outlist[4]
            if spk not in spkdict.keys():
                spkdict[spk] = spkid
                spkid += 1
            outlist[4] = str(spkid)
            
            outline = '|'.join(outlist) + '\n'
            opsubt.write(outline)
            if allsave_flag:
                oallt.write(outline)

            tmp_idx += 1
    with open(keys_file, 'w') as okf:
        save_keys = ['uttid']
        save_keys.extend(all_keylist)
        oline = '|'.join(save_keys) + '\n'
        okf.write(oline)

    if allsave_flag:
        oallt.close()
    opsubt.close()
    olistsub.close()
    with open(spk_id_file, 'w', encoding="utf-8") as opf:
        json.dump(spkdict, opf, indent=2, ensure_ascii=False)
    
        
