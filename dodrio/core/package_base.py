
import os
import numpy as np
import librosa
from tqdm import tqdm
import glob
from io import BytesIO
import json
from scipy.io import wavfile
import pyarrow.parquet as pq


############## Parquet to Package ###############   

def audio_regular(audio, ori_sample_rate, dtype, target_sample_rate=48000):
    '''
    descripttion: 
    param {*} audio
    param {*} ori_sample_rate
    param {str} dtype : 'int16' or 'int32'
    param {int} target_sample_rate : target sample rate
    return {*}
    '''
    if dtype == 'int16':
        dnum = 32768
    elif dtype == 'int32':
        dnum = 2147483648
    else:
        print(f"{dtype} is not normal data type")
    norm_audio = audio / dnum
    rs_audio = librosa.resample(norm_audio, orig_sr=ori_sample_rate, target_sr=target_sample_rate)
    int16_audio = (rs_audio*32768).astype(np.int16)
    return int16_audio

def load_mp3(bio, target_sample_rate=48000):
    audio, ori_sr = librosa.load(BytesIO(bio))
    rs_audio = librosa.resample(audio, orig_sr=ori_sr, target_sr=target_sample_rate)
    int16_audio = (rs_audio*32768).astype(np.int16)
    return int16_audio

def save_meta_info(outdir, sample_rate, num_utts_per_parquet):
    outpath = os.path.join(outdir, 'meta_info.json')
    outinfo = {
        'sample_rate': sample_rate,
        'num_utts_per_parquet': num_utts_per_parquet,
    }
    with open(outpath, 'w', encoding="utf-8") as opf:
        json.dump(outinfo, opf, indent=2, ensure_ascii=False)

def parquet2package_single(parquet_file, package_file, target_sample_rate=48000):
    ftype = os.path.split(parquet_file)[-1].split('_')[0]
    df = pq.read_table(parquet_file).to_pandas()
    basename = os.path.split(parquet_file)[-1].split('.parquet')[0]
    position = 0
    info_list = []
    audio_pos = {}
    outf = open(package_file, 'wb')
    for idx in tqdm(range(len(df)), desc=f'{basename} Processing'):
        utt = df.iloc[idx]['utt']
        sr = df.iloc[idx]['sample_rate']
        dtype = df.iloc[idx]['dtype']
        audio = df.iloc[idx]['audio_data']
        if ftype == 'wav':
            reg_audio = audio_regular(audio, sr, dtype, target_sample_rate)
        elif ftype == 'mp3':
            reg_audio = load_mp3(audio, target_sample_rate) 
        else:
            print("Now just accept mp3 and wav format")
            return

        byte_audio = bytes(reg_audio)
        outf.write(byte_audio)

        byte_num = len(reg_audio)* 2 
        end_position = position+byte_num 
        audio_pos[utt] = [position, end_position]
        info_list.append([utt, os.path.split(package_file)[-1], str(position), str(end_position)])

        position += byte_num
    return info_list

def parquet2package(parquet_dir, package_dir, sample_rate=48000):
    os.makedirs(package_dir, exist_ok=True)
    pq_file_list = glob.glob(parquet_dir+'/*.parquet')
    pq_file_list.sort()
    all_info_list = []
    num_utts_per_parquet = len(pq_file_list)
    save_meta_info(package_dir, sample_rate, num_utts_per_parquet)
    for one_pq in pq_file_list:
        basename = os.path.split(one_pq)[-1].split('.parquet')[0]
        ftype = os.path.split(one_pq)[-1].split('_')[0]
        pack_file = os.path.join(package_dir, basename+'.pack')
        info_list = parquet2package_single(one_pq, pack_file, sample_rate)
        all_info_list.append(info_list)
    info_outfile = os.path.join(package_dir, 'uttinfo.list')
    with open(info_outfile, 'w') as outf:
        for info_list in all_info_list:
            for info in info_list:
                outline = '|'.join(info) + '\n'
                outf.write(outline)

############## Package to Wav ############### 

def load_data_from_package(pack_file, start, end):
    with open(pack_file, 'rb') as pf :
        pf.seek(start)
        data = pf.read(end-start)
    audio = np.frombuffer(data, dtype=np.int16)
    return audio
def save_wav(audio, wavpath, sr):
    wavfile.write(wavpath, sr, audio)
    
def package2wav(package_dir, wav_dir):
    os.makedirs(wav_dir, exist_ok=True)
    info_file = os.path.join(package_dir, 'uttinfo.list')
    meta_file = os.path.join(package_dir, 'meta_info.json')
    with open(meta_file, 'r') as fm:
        minfo = json.load(fm)
        sample_rate = minfo['sample_rate']
    with open(info_file, 'r') as iff:
        info_list = iff.readlines()
    for info in tqdm(info_list):
        utt, pf, start, end = info.split('|')
        start = int(start)
        end = int(end)
        wpf = os.path.join(package_dir, pf)
        audio = load_data_from_package(wpf, start, end)
        wavpath = os.path.join(wav_dir, utt+'.wav')
        save_wav(audio, wavpath, sample_rate)