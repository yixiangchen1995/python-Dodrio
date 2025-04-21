'''
FilePath: /python-Dodrio/dodrio/afeat/exp_load.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-04-14 16:56:12
LastEditors: Yixiang Chen
LastEditTime: 2025-04-15 15:19:07
'''

import json
import os

def get_file_list(inp_dir, suffix='.wav'):
    itm = []
    for home, dirs, files in os.walk(inp_dir):
        pppp = list( map(lambda fname: home + '/' + fname,
            list( filter( lambda filename: os.path.splitext(filename)[1] == suffix,
            files) ) ) )
        itm.extend(pppp)
    file_list = itm
    return file_list

def load_mfajson(jsonfile):
    with open(jsonfile, 'r') as af:
        adict = json.load(af)
    word_align_list = adict['tiers']['words']['entries']
    phone_align_list = adict['tiers']['phones']['entries']

    wstart = [ww[0]*1000 for ww in word_align_list]
    wend = [ww[1]*1000 for ww in word_align_list] 
    wlist = [ww[2] for ww in word_align_list]

    pstart = [pp[0]*1000 for pp in phone_align_list]
    pend = [pp[1]*1000 for pp in phone_align_list]
    plist = [pp[2] for pp in phone_align_list]

    pdur = [int(int(pend[idx])-int(pstart[idx])) for idx in range(len(pstart))]

    return wstart, wend, wlist, pstart, pend, plist, pdur

def get_mfa(wav, utt, filedict):
    jsonfile = filedict[utt]
    wstart, wend, wlist, pstart, pend, plist, pdur = load_mfajson(jsonfile)
    return wstart, wend, wlist, pstart, pend, plist, pdur 

def align_jsondict(jsondir):
    jsonflist = get_file_list(jsondir, '.json')
    jsonfiledict = {}
    for jsonf in jsonflist:
        basename = os.path.split(jsonf)[-1].split('.json')[0]
        jsonfiledict[basename] = jsonf
    return jsonfiledict 

def align_jsondict_aa(jsondir):
    jsonfiledict = {}
    eplist = os.listdir(jsondir)
    for epd in eplist:
        aligndir = os.path.join(jsondir, epd, 'align_new')
        jsonflist = get_file_list(aligndir, '.json')
        for jsonf in jsonflist:
            basename = os.path.split(jsonf)[-1].split('.json')[0]
            jsonfiledict[basename] = jsonf
    return jsonfiledict

def align_jsondict_exc(jsondir):
    jsonfiledict = {}
    aligndir = os.path.join(jsondir, 'align_new')
    jsonflist = get_file_list(aligndir, '.json')
    for jsonf in jsonflist:
        basename = os.path.split(jsonf)[-1].split('.json')[0]
        basename = '_'.join(basename.split('_')[1:])
        jsonfiledict[basename] = jsonf
    return jsonfiledict
