'''
FilePath: /python-Dodrio/dodrio/afeat/feat_extractor.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-26 19:12:18
LastEditors: Yixiang Chen
LastEditTime: 2025-04-21 15:55:04
'''


import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from dodrio.tools.load_data import load_data_dict, load_pack_audio_data

from dodrio.afeat.exp_load import align_jsondict, align_jsondict_aa, align_jsondict_exc


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
    packlist.sort()
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


def save_parquet_align(wavinfo_dict, savelist, parquet_fn):
    word_start = [wavinfo_dict[x][0] for x in savelist]
    word_end = [wavinfo_dict[x][1] for x in savelist]
    word_list = [wavinfo_dict[x][2] for x in savelist]

    phone_start = [wavinfo_dict[x][3] for x in savelist]
    phone_end = [wavinfo_dict[x][4] for x in savelist]
    phone_list = [wavinfo_dict[x][5] for x in savelist]
    phone_duration = [wavinfo_dict[x][6] for x in savelist] 
    df = pd.DataFrame()
    df['utt'] = savelist
    df['word_start'] = word_start
    df['word_end'] = word_end
    df['word_list'] = word_list
    df['phone_start'] = phone_start
    df['phone_end'] = phone_end
    df['phone_list'] = phone_list
    df['phone_duration'] = phone_duration
    df.to_parquet(parquet_fn)
    #print(f"{parquet_fn} had be saved")

def extract_feat_align(extractor_func, featname, input_dir, out_dir, from_type, jsondir, **params):
    os.makedirs(out_dir, exist_ok=True)
    #data_dict = load_data_dict(input_dir, from_type)
    if from_type=='parquet':
        suffix = '.parquet'
        infolistf = ''
    else:
        suffix = '.pack'
        infolistf = os.path.join(input_dir, 'uttinfo.list')
    packlist = get_file_list(input_dir, suffix)
    packlist.sort()
    feat_info_file = os.path.join(out_dir, 'feat_info_'+featname+'.list')
    oinfo_out = open(feat_info_file, 'w')

    jsonfiledict = align_jsondict_aa(jsondir)
    #jsonfiledict = align_jsondict_exc(jsondir)

    for idx in range(len(packlist)):
        packfile = packlist[idx]
        packid = os.path.split(packfile)[-1].split(suffix)[0]
        uttdict = load_pack_audio_data(packfile, infolistf, return_sr = False)
        
        out_feat_path = os.path.join(out_dir, packid+'.'+featname)
        info_dict = {}
        for utt in tqdm(uttdict.keys(), desc=f'{packid} Processing'): 
            wavdata = uttdict[utt]
            if utt not in jsonfiledict.keys():
                continue
            wstart, wend, wlist, pstart, pend, plist, pdur = extractor_func(wavdata, utt, jsonfiledict, **params)

            info_dict[utt] = [wstart, wend, wlist, pstart, pend, plist, pdur]

            feat_info = [utt, os.path.split(out_feat_path)[-1], ' '.join(plist), ' '.join([str(xx) for xx in pdur]) ]
            info_outline = '|'.join(feat_info) + '\n'
            oinfo_out.write(info_outline) 

        save_parquet_align(info_dict, info_dict.keys(), out_feat_path)
    oinfo_out.close()

def extract_feat_multi(extractor_func, featname_list, input_dir, out_dir_list, from_type, **params):

    if from_type=='parquet':
        suffix = '.parquet'
        infolistf = ''
    else:
        suffix = '.pack'
        infolistf = os.path.join(input_dir, 'uttinfo.list')
    packlist = get_file_list(input_dir, suffix)
    packlist.sort()

    oinfo_out_list = []
    for ii in range(len(featname_list)):
        out_dir = out_dir_list[ii]
        featname = featname_list[ii]
        os.makedirs(out_dir, exist_ok=True)
        feat_info_file = os.path.join(out_dir, 'feat_info_'+featname+'.list')
        oinfo_out = open(feat_info_file, 'w')
        oinfo_out_list.append(oinfo_out)

    for idx in range(len(packlist)):
        packfile = packlist[idx]
        packid = os.path.split(packfile)[-1].split(suffix)[0]
        uttdict = load_pack_audio_data(packfile, infolistf, return_sr = False)
        
        positions = []
        outf_list = []
        out_feat_path_list = []
        for ii in range(len(featname_list)):
            positions.append(0)
            featname = featname_list[ii]
            out_dir = out_dir_list[ii]
            out_feat_path = os.path.join(out_dir, packid+'.'+featname)
            out_feat_path_list.append(out_feat_path)
            outf = open(out_feat_path, 'wb')
            outf_list.append(outf)

        for utt in tqdm(uttdict.keys(), desc=f'{packid} Processing'): 
            wavdata = uttdict[utt]
            feat_list = extractor_func(wavdata, utt, **params)
            for ii in range(len(feat_list)):
                feat = feat_list[ii]
                if feat is None:
                    feat = np.array([0]).astype(np.float32) # shape is 1
                feat = feat.astype(np.float32)
                fshape = feat.shape
                feat = np.reshape(feat, -1)

                byte_feat = bytes(feat)
                outf_list[ii].write(byte_feat)

                byte_num = len(feat)* 4 # float 32 = 4 byte 
                end_position = positions[ii]+byte_num 
                #feat_info = [utt, os.path.split(out_feat_path)[-1], str(positions[ii]), str(end_position), ','.join([str(xx) for xx in fshape])]
                feat_info = [utt, os.path.split(out_feat_path_list[ii])[-1], str(positions[ii]), str(end_position), ','.join([str(xx) for xx in fshape])]
                info_outline = '|'.join(feat_info) + '\n'
                oinfo_out_list[ii].write(info_outline) 

                positions[ii] += byte_num
        for outf in outf_list:
            outf.close()
    for oinfo_out in oinfo_out_list:
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


