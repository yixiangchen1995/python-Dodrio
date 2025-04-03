'''
FilePath: /python-Dodrio/dodrio/afeat/feat_extractor.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-26 19:12:18
LastEditors: Yixiang Chen
LastEditTime: 2025-04-03 17:31:42
'''


import os
from tqdm import tqdm
import numpy as np

from dodrio.tools.load_data import load_data_dict, load_pack_audio_data

def get_file_list(inp_dir, suffix='.wav'):
    itm = []
    for home, dirs, files in os.walk(inp_dir):
        pppp = list( map(lambda fname: home + '/' + fname,
            list( filter( lambda filename: os.path.splitext(filename)[1] == suffix,
            files) ) ) )
        itm.extend(pppp)
    file_list = itm
    return file_list

def extract_feat_allload(extractor_func, featname, input_dir, out_dir, from_type, **params):
    os.makedirs(out_dir, exist_ok=True)
    data_dict = load_data_dict(input_dir, from_type)
    feat_info_file = os.path.join(out_dir, 'feat_info_'+featname+'.list')
    oinfo_out = open(feat_info_file, 'w')
    for packid in data_dict.keys():
        uttdict = data_dict[packid]
        out_feat_path = os.path.join(out_dir, packid+'.'+featname)
        outf = open(out_feat_path, 'wb')
        position = 0
        for utt in tqdm(uttdict.keys(), desc=f'{packid} Processing'): 
            wavdata = uttdict[utt]
            feat = extractor_func(wavdata, utt, **params)
            if feat is None:
                feat = np.array([0]).astype(np.float32) # shape is 1
            feat = feat.astype(np.float32)
            fshape = feat.shape
            feat = np.reshape(feat, -1)

            byte_feat = bytes(feat)
            outf.write(byte_feat)

            byte_num = len(feat)* 4 # float 32 = 4 byte 
            end_position = position+byte_num 
            feat_info = [utt, os.path.split(out_feat_path)[-1], str(position), str(end_position), ','.join([str(xx) for xx in fshape])]
            info_outline = '|'.join(feat_info) + '\n'
            oinfo_out.write(info_outline) 

            position += byte_num
        outf.close()
    oinfo_out.close()

def extract_feat(extractor_func, featname, input_dir, out_dir, from_type, **params):
    os.makedirs(out_dir, exist_ok=True)
    #data_dict = load_data_dict(input_dir, from_type)
    if from_type=='parquet':
        suffix = '.parquet'
        infolistf = ''
    else:
        suffix = '.pack'
        infolistf = os.path.join(input_dir, 'uttinfo.list')
    packlist = get_file_list(input_dir, suffix)
    feat_info_file = os.path.join(out_dir, 'feat_info_'+featname+'.list')
    oinfo_out = open(feat_info_file, 'w')

    for idx in range(len(packlist)):
        packfile = packlist[idx]
        packid = os.path.split(packfile)[-1].split(suffix)[0]
        uttdict = load_pack_audio_data(packfile, infolistf, return_sr = False)
        
        out_feat_path = os.path.join(out_dir, packid+'.'+featname)
        outf = open(out_feat_path, 'wb')
        position = 0
        for utt in tqdm(uttdict.keys(), desc=f'{packid} Processing'): 
            wavdata = uttdict[utt]
            feat = extractor_func(wavdata, utt, **params)
            if feat is None:
                feat = np.array([0]).astype(np.float32) # shape is 1
            feat = feat.astype(np.float32)
            fshape = feat.shape
            feat = np.reshape(feat, -1)

            byte_feat = bytes(feat)
            outf.write(byte_feat)

            byte_num = len(feat)* 4 # float 32 = 4 byte 
            end_position = position+byte_num 
            feat_info = [utt, os.path.split(out_feat_path)[-1], str(position), str(end_position), ','.join([str(xx) for xx in fshape])]
            info_outline = '|'.join(feat_info) + '\n'
            oinfo_out.write(info_outline) 

            position += byte_num
        outf.close()
    oinfo_out.close()


def get_utt2spk(infodir):
    utt2spk = {}
    inpfile = os.path.join(infodir, 'uttinfo_text.list')
    with open(inpfile, 'r') as oif:
        info_lines = oif.readlines()
    index_list = info_lines[0].strip().split('|')
    idid, spkid = -1, -1 
    for idx in range(len(index_list)):
        if index_list[idx] == 'id':
            idid = idx
        if index_list[idx] == 'speaker':
            spkid = idx
    for idx in range(1, len(info_lines)):
        spl = info_lines[idx].strip().split('|')
        utt = spl[idid]
        spk = spl[spkid]
        utt2spk[utt] = spk
    del info_lines
    return utt2spk


