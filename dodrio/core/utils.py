'''
FilePath: /base-dodrio/dodrio/core/utils.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 11:51:05
LastEditors: Yixiang Chen
LastEditTime: 2025-03-24 16:56:54
'''

import os
from tqdm import tqdm

def get_file_list(inp_dir, suffix='.wav'):
    itm = []
    for home, dirs, files in os.walk(inp_dir):
        pppp = list( map(lambda fname: home + '/' + fname,
            list( filter( lambda filename: os.path.splitext(filename)[1] == suffix,
            files) ) ) )
        itm.extend(pppp)
    file_list = itm
    return file_list

def get_file_list_tail(inp_dir, suffix='_text.txt'):
    itm = []
    for home, dirs, files in os.walk(inp_dir):
        pppp = list( map(lambda fname: home + '/' + fname,
            list( filter( lambda filename: filename[-len(suffix):] == suffix,
            files) ) ) )
        itm.extend(pppp)
    file_list = itm
    return file_list


def utt_name_tran(basename, rm_prefix=False):
    #return basename
    if rm_prefix: 
        return '_'.join(basename.split('_')[1:])
    else:
        return basename

def set_wavlist(wav_dir, file_type, rm_prefix=False):
    suffix = '.'+file_type
    wavlist = get_file_list(wav_dir, suffix)
    wavdict = {}
    uttlist = []
    for wavpath in tqdm(wavlist, desc='SetList'):
        (path, filename) = os.path.split(wavpath)
        basename = filename.split(suffix)[0]
        uttname = utt_name_tran(basename, rm_prefix)
        wavdict[uttname] = wavpath
        uttlist.append(uttname)
    return wavdict, uttlist

