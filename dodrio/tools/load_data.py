'''
FilePath: /python-Dodrio/dodrio/tools/load_data.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-26 19:23:27
LastEditors: Yixiang Chen
LastEditTime: 2026-06-26 17:39:52
'''


import os
from io import BytesIO
import librosa
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq


####################### Load data block ###################### 

def get_file_list(inp_dir, suffix='.wav'):
    itm = []
    for home, dirs, files in os.walk(inp_dir):
        pppp = list( map(lambda fname: home + '/' + fname,
            list( filter( lambda filename: os.path.splitext(filename)[1] == suffix,
            files) ) ) )
        itm.extend(pppp)
    file_list = itm
    return file_list

def load_datalist_dict(input_dir, from_type='parquet'):
    if from_type == 'parquet':
        load_file = os.path.join(input_dir, 'utt2parquet.list')
    else:
        load_file = os.path.join(input_dir, 'uttinfo.list')
    data_dict = {}
    with open(load_file, 'r') as lf:
        lines = lf.readlines()
    for line in lines:
        spl = line.strip().split('|')
        name= spl[0]
        packname = spl[1].split('.')[0]
        data_dict.setdefault(packname, []).append(name)
    return data_dict

def load_data_dict(input_dir, from_type='parquet'):
    if from_type=='parquet':
        suffix = '.parquet'
        infolistf = ''
    else:
        suffix = '.pack'
        infolistf = os.path.join(input_dir, 'uttinfo.list')
    packlist = get_file_list(input_dir, suffix)
    packlist.sort()
    data_dict = {}
    for idx in range(len(packlist)):
        packfile = packlist[idx]
        basename = os.path.split(packfile)[-1].split(suffix)[0]
        uttdict = load_pack_audio_data(packfile, infolistf, return_sr = False)
        data_dict[basename] = uttdict
    return data_dict


####################### Load audio ######################  

def audiotran(data, atype, dtype):
    if atype == 'wav':
        if dtype == 'int16':
            dnum = 32768
        elif dtype == 'int32':
            dnum = 2147483648
        else:
            print(f"{dtype} is not normal data type")
            return 
        return data / dnum
    elif atype == 'mp3':
        wav, sr = librosa.load(BytesIO(data))
        return wav

def load_pack_audio_data_old(packp, infolistf='', return_sr = False):
    ftype = os.path.split(packp)[-1].split('.')[-1]
    assert ftype in ['parquet', 'pack']
    audio_type = os.path.split(packp)[-1].split('_')[0]
    assert audio_type in ['wav', 'mp3']
    outdict = {}
    if ftype == 'parquet':
        df = pq.read_table(packp).to_pandas()
        for idx in tqdm(range(len(df)), desc=f'Loading'):
            utt = df.iloc[idx]['utt']
            sr = df.iloc[idx]['sample_rate']
            dtype = df.iloc[idx]['dtype']
            audio = df.iloc[idx]['audio_data']
            wav = audiotran(audio, audio_type, dtype)
            if return_sr:
                outdict[utt] = [wav,sr]
            else:
                outdict[utt] = wav
    else:
        info_file = os.path.join(infolistf)
        with open(info_file, 'r') as iff:
            info_list = iff.readlines()
        packname = os.path.split(packp)[-1] 
        #for info in tqdm(info_list):
        for info in info_list:
            utt, pf, start, end = info.split('|')
            start = int(start)
            end = int(end)
            if pf != packname:
                continue
            with open(packp, 'rb') as opf :
                opf.seek(start)
                data = opf.read(end-start)
            wav = np.frombuffer(data, dtype=np.int16)
            wav = wav / 32768
            if return_sr:
                outdict[utt] = [wav,48000]
            else:
                outdict[utt] = wav
    return outdict

def load_pack_audio_data(packp, infolistf='', return_sr=False):
    ftype = os.path.split(packp)[-1].split('.')[-1]
    assert ftype in ['parquet', 'pack'], f"Unsupported file type: {ftype}"
    
    audio_type = os.path.split(packp)[-1].split('_')[0]
    assert audio_type in ['wav', 'mp3'], f"Unsupported audio type: {audio_type}"
    
    outdict = {}
    
    if ftype == 'parquet':
        df = pq.read_table(packp).to_pandas()
        for idx in tqdm(range(len(df)), desc=f'Loading parquet'):
            utt = df.iloc[idx]['utt']
            sr = df.iloc[idx]['sample_rate']
            dtype = df.iloc[idx]['dtype']
            audio = df.iloc[idx]['audio_data']
            wav = audiotran(audio, audio_type, dtype)
            if return_sr:
                outdict[utt] = [wav, sr]
            else:
                outdict[utt] = wav
    else:
        # --- 优化开始：针对 .pack 文件 ---
        
        # 1. 读取 info_list
        info_file = os.path.join(infolistf)
        with open(info_file, 'r') as iff:
            info_list = iff.readlines()
            
        packname = os.path.split(packp)[-1]
        
        # 2. 【核心优化】一次性将整个 pack 文件读入内存
        # 使用 'rb' 模式读取所有字节
        with open(packp, 'rb') as f:
            full_pack_data = f.read()
            
        # 获取文件大小用于可选的边界检查（防止 info 中的 end 超出文件实际大小）
        file_size = len(full_pack_data)

        # 3. 在内存中根据 info_list 提取数据
        # 注意：如果 info_list 非常大，可以考虑先过滤出属于当前 packname 的行，减少循环次数
        # 这里保持原有逻辑结构，但去除了内部的文件 I/O
        
        #for info in tqdm(info_list, desc=f'Loading pack {packname}'):
        for info in info_list:
            info = info.strip() # 去除可能的换行符
            if not info:
                continue
                
            parts = info.split('|')
            if len(parts) != 4:
                continue # 跳过格式错误的行
                
            utt, pf, start_str, end_str = parts
            
            # 只处理属于当前 pack 文件的条目
            if pf != packname:
                continue
            
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue # 跳过无法解析坐标的行

            # 边界检查（可选，但推荐）
            if start < 0 or end > file_size or start >= end:
                # print(f"Warning: Invalid range for {utt}: [{start}, {end}] in file of size {file_size}")
                continue

            # 3.1 【核心优化】从内存 bytes 中切片，而不是从磁盘读取
            # data 是一个 bytes 对象
            data = full_pack_data[start:end]
            
            # 3.2 转换为 numpy 数组
            # 注意：np.frombuffer 返回的是只读数组，如果需要修改 wav，建议 copy()
            # 原代码没有 copy，这里保持一致。如果后续有 inplace 操作，请改为 np.frombuffer(data, dtype=np.int16).copy()
            wav = np.frombuffer(data, dtype=np.int16)
            
            # 归一化
            wav = wav / 32768.0
            
            if return_sr:
                outdict[utt] = [wav, 48000]
            else:
                outdict[utt] = wav
                
        # --- 优化结束 ---

    return outdict

######################## Load Text #########################
def load_textinfo_data(infolist):
    with open(infolist, 'r') as oifl:
        lines = oifl.readlines()
    keys = lines[0].strip().split('|')
    id_idx = 0
    for idx in range(len(keys)):
        if keys[idx] == 'id':
            id_idx = idx
            break
    info_dict = {}
    for idx in range(1, len(lines)):
        line = lines[idx]
        spl = line.strip().split('|')
        info_dict[spl[id_idx]] = {}
        for idx in range(len(keys)):
            kk = keys[idx]
            info_dict[spl[id_idx]][kk] = spl[idx]
    return info_dict

####################### Load single ######################

def load_audio_single(packp, start, end):
    with open(packp, 'rb') as opf :
        opf.seek(start)
        data = opf.read(end-start)
    wav = np.frombuffer(data, dtype=np.int16)
    wav = wav / 32768
    return wav

def load_feat_single(featpack, start, end, shape):
    with open(featpack, 'rb') as opf :
        opf.seek(start)
        data = opf.read(end-start)
    feat = np.frombuffer(data, dtype=np.float32)
    feat = np.reshape(feat, shape)
    return feat

def load_data_from_line(infoline):
    spl = infoline.strip().split('|')
    data_dict = {}
    uttid = spl[0]
    wavp, wstart, wend = spl[1], int(spl[2]), int(spl[3])
    audio = load_audio_single(wavp, wstart, wend) 
    spkid, text, language = spl[4], spl[5], spl[6]

    data_dict['uttid'] = uttid
    data_dict['audio'] = audio
    data_dict['spkid'] = spkid
    data_dict['text'] = text
    data_dict['language'] = language

    feat_type_num = (len(spl)- 7) // 5
    for idx in range(feat_type_num):
        beginid = 7 + idx*5 
        featname = spl[beginid]
        featpack = spl[beginid+1]
        fstart, fend, fshape = int(spl[beginid+2]), int(spl[beginid+3]), spl[beginid+4]
        fshape = [int(xx) for xx in fshape.split(',')]
        feat_data = load_feat_single(featpack, fstart, fend, fshape)
        data_dict[featname] = feat_data

    return data_dict 
        
def load_data_from_line_align(infoline):
    spl = infoline.strip().split('|')
    data_dict = {}
    uttid = spl[0]
    wavp, wstart, wend = spl[1], int(spl[2]), int(spl[3])
    audio = load_audio_single(wavp, wstart, wend) 
    spkid, text, language = spl[4], spl[5], spl[6]

    phones, duration = spl[7], spl[8]
    phones = phones.split()
    duration = [int(xx) for xx in duration.split()]

    data_dict['uttid'] = uttid
    data_dict['audio'] = audio
    data_dict['spkid'] = spkid
    data_dict['text'] = text
    data_dict['language'] = language
    data_dict['phones'] = phones
    data_dict['duration'] = duration

    feat_type_num = (len(spl)- 9) // 5
    for idx in range(feat_type_num):
        beginid = 9 + idx*5 
        featname = spl[beginid]
        featpack = spl[beginid+1]
        fstart, fend, fshape = int(spl[beginid+2]), int(spl[beginid+3]), spl[beginid+4]
        fshape = [int(xx) for xx in fshape.split(',')]
        feat_data = load_feat_single(featpack, fstart, fend, fshape)
        data_dict[featname] = feat_data

    return data_dict 


