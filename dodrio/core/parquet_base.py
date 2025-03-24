'''
FilePath: /base-dodrio/dodrio/core/parquet_base.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 11:26:49
LastEditors: Yixiang Chen
LastEditTime: 2025-03-24 16:57:57
'''

import os
import pandas as pd
from tqdm import tqdm
import math
import multiprocessing
import glob
from scipy.io import wavfile
import pyarrow.parquet as pq

from tinytag import TinyTag # for read mp3 metainfo

from dodrio.core.utils import set_wavlist

############## Wav to Parquet ###############

def save_parquet(wavinfo_dict, savelist, parquet_fn):
    sr_list = [wavinfo_dict[x][0] for x in savelist]
    dtype_list = [wavinfo_dict[x][1] for x in savelist]
    audio_list = [wavinfo_dict[x][2] for x in savelist]
    df = pd.DataFrame()
    df['utt'] = savelist
    df['sample_rate'] = sr_list
    df['dtype'] = dtype_list
    df['audio_data'] = audio_list
    df.to_parquet(parquet_fn)
    print(f"{parquet_fn} had be saved")

def get_mp3_metainfo(mp3file):
    tag = TinyTag.get(mp3file)
    return tag.samplerate

def gen_parquet(wav_dir, parquet_dir, mid_name='', file_type='wav', num_utts_per_parquet=2000, num_processes=5, process_max_num=10000):
    os.makedirs(parquet_dir, exist_ok=True)
    wavdict, uttlist = set_wavlist(wav_dir, file_type)

    parquet2utt = {}
    turn_num = math.ceil(len(uttlist) / process_max_num)
    for tid in range(turn_num):
        wavinfo_dict ={}
        for utt in tqdm(uttlist[process_max_num * tid : process_max_num * (tid+1)], desc=f'Turn{str(tid)}LoadAudio'):
            wavpath = wavdict[utt]
            if file_type == 'wav':
                sr, wav = wavfile.read(wavpath)
                if len(wav)<1:
                    uttlist.remove(utt)
                    print(f"{utt} wavfile is None")
                    continue
                if len(wav.shape) > 1:
                    print(f"{utt} {str(len(wav.shape))} channel is non-mono channel, just first channel will be save")
                    wav = wav[:,0]
                dtype = str(wav.dtype)
                wavinfo_dict[utt] = [sr, dtype, wav]
            elif file_type == 'mp3':
                byte_mp3_data = open(wavpath, 'rb').read()
                sr = get_mp3_metainfo(wavpath)
                dtype = 'mp3'
                wavinfo_dict[utt] = [sr, dtype, byte_mp3_data]
            else:
                print("Now just accept mp3 and wav format")
                return

        # Using process pool to speedup
        prefix = file_type
        pool = multiprocessing.Pool(processes=num_processes)
        parquet_list = []
        for i, j in enumerate(range(process_max_num * tid, min(len(uttlist), process_max_num * (tid+1)), num_utts_per_parquet)):
            idx = (process_max_num * tid) // num_utts_per_parquet + i
            pfile = prefix + '_' + mid_name + '_{:05d}.parquet'.format(idx)
            parquet_file = os.path.join(parquet_dir, pfile)
            parquet_list.append(parquet_file)
            parquet2utt[pfile] = uttlist[j: j + num_utts_per_parquet] 
            pool.apply_async(save_parquet, (wavinfo_dict, uttlist[j: j + num_utts_per_parquet], parquet_file))
        pool.close()
        pool.join()

    utt2parquet_file = os.path.join(parquet_dir, 'utt2parquet.list') 
    u2pf = open(utt2parquet_file, 'w')
    for pak in parquet2utt.keys():
        for utt in parquet2utt[pak]:
            outline = utt+'|' + pak + '\n'
            u2pf.write(outline)
    u2pf.close()


############## Parquet to Wav ###############

def parquet2wav_single(parquet_file, wav_dir): 
    ftype = os.path.split(parquet_file)[-1].split('_')[0]
    df = pq.read_table(parquet_file).to_pandas()
    basename = os.path.split(parquet_file)[-1].split('.parquet')[0]
    for idx in tqdm(range(len(df)), desc=f'{basename} Processing'):
        utt = df.iloc[idx]['utt']
        sr = df.iloc[idx]['sample_rate']
        dtype = df.iloc[idx]['dtype']
        audio = df.iloc[idx]['audio_data']
        if ftype == 'wav':
            wavpath = os.path.join(wav_dir, utt+'.wav')
            wavfile.write(wavpath, sr, audio)
        else:
            wavpath = os.path.join(wav_dir, utt+'.'+ftype)
            with open(wavpath, 'wb') as ww:
                ww.write(audio)

def parquet2wav(parquet_dir, wav_dir):
    os.makedirs(wav_dir, exist_ok=True)
    pq_file_list = glob.glob(parquet_dir+'/*.parquet')
    for parquet_file in pq_file_list:
        parquet2wav_single(parquet_file, wav_dir)