'''
FilePath: /python-Dodrio/dodrio/core/parquet_base.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 11:26:49
LastEditors: Yixiang Chen
LastEditTime: 2026-06-29 17:04:02
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

from dodrio.core.utils import set_wavlist, set_wavlist_predir, set_wavlist_dirlist

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

def save_parquet_oss_version(wavinfo_dict, savelist, parquet_fn):
    #print("savelist is ", savelist)
    sr_list = [wavinfo_dict[x][0] for x in savelist]
    dtype_list = [wavinfo_dict[x][1] for x in savelist]
    audio_list = [wavinfo_dict[x][2] for x in savelist]
    df = pd.DataFrame()
    df['utt'] = savelist
    df['sample_rate'] = sr_list
    df['dtype'] = dtype_list
    df['audio_data'] = audio_list
    #df.to_parquet(parquet_fn)
    from io import BytesIO
    buf = BytesIO()
    df.to_parquet(buf, engine='pyarrow', index=False)
    data = buf.getvalue()
    with open(parquet_fn, 'wb') as f:
        f.write(data)
    print(f"{parquet_fn} had be saved")


def get_mp3_metainfo(mp3file):
    tag = TinyTag.get(mp3file)
    return tag.samplerate

def load_single_audio(utt, wavpath, file_type):
    """
    单个音频加载函数，用于多线程调用
    返回: (utt, data_info) 或 None (如果失败)
    """
    if file_type == 'wav':
        try:
            sr, wav = wavfile.read(wavpath)
        except FileNotFoundError:
            print(f"⚠️ 文件不存在，跳过: {wavpath}")
            return None
        except ValueError:
            print(f"⚠️ 文件格式错误或损坏，跳过: {wavpath}")
            return None
        except Exception as e:
            print(f"❌ 未知错误，跳过: {wavpath} | 错误类型: {type(e).__name__}, 详情: {e}")
            return None
        
        if len(wav) < 1:
            print(f"{utt} wavfile is None")
            return None
        if len(wav.shape) > 1:
            print(f"{utt} {str(len(wav.shape))} channel is non-mono channel, just first channel will be save")
            wav = wav[:, 0]
        dtype = str(wav.dtype)
        return utt, [sr, dtype, wav]
        
    elif file_type == 'mp3':
        try:
            # 注意：open().read() 也是 IO 操作，多线程有效
            byte_mp3_data = open(wavpath, 'rb').read()
            sr = get_mp3_metainfo(wavpath)
            return utt, [sr, 'mp3', byte_mp3_data]
        except Exception:
            return None
    else:
        return None


from concurrent.futures import ThreadPoolExecutor, as_completed

def gen_parquet(wav_dir, parquet_dir, mid_name='', file_type='wav', 
                num_utts_per_parquet=2000, num_save_processes=5, 
                process_max_num=10000, use_pre_dir=False, pre_dir='merge',
                use_dirlist=False, csv_name='result_step2.csv', num_load_threads=16): # 新增：读取线程数
    """
    优化版：
    1. 外层 Turn 串行（控制内存）。
    2. 内层 Load 使用多线程并行（加速 IO/解码）。
    3. 内层 Save 使用多进程并行（加速 CPU/磁盘写入）。
    """
    os.makedirs(parquet_dir, exist_ok=True)
    
    if use_dirlist:
        wavdict, uttlist = set_wavlist_dirlist(wav_dir, file_type, csv_name=csv_name)
    elif use_pre_dir:
        wavdict, uttlist = set_wavlist_predir(wav_dir, file_type, pre_dir=pre_dir)
    else:
        wavdict, uttlist = set_wavlist(wav_dir, file_type)

    total_utts = len(uttlist)
    turn_num = math.ceil(total_utts / process_max_num)
    
    # 用于最后生成映射文件
    final_parquet2utt = {}

    print(f"总文件数: {total_utts}, 分 {turn_num} 个 Turns 处理")
    print(f"配置: 加载线程数={num_load_threads}, 保存进程数={num_save_processes}")

    #for tid in tqdm.tqdm(range(turn_num), desc="Overall Progress"):
    for tid in tqdm(range(turn_num), desc="Overall Progress"):
        start_idx = process_max_num * tid
        end_idx = min(total_utts, process_max_num * (tid + 1))
        current_turn_utts = uttlist[start_idx:end_idx]
        
        wavinfo_dict = {}
        
        # ================= 核心优化点：并行加载 =================
        # 准备任务列表
        load_tasks = [(utt, wavdict[utt], file_type) for utt in current_turn_utts]
        
        # 使用线程池并行读取
        # max_workers: 如果是 SSD，可以设高一点（如 32-64）；如果是 HDD，设低一点（如 8-16）
        with ThreadPoolExecutor(max_workers=num_load_threads) as executor:
            # 提交所有任务
            future_to_utt = {executor.submit(load_single_audio, utt, path, file_type): utt 
                             for utt, path, _ in load_tasks}
            
            # 获取结果
            #for future in tqdm.tqdm(as_completed(future_to_utt), total=len(load_tasks), 
            for future in tqdm(as_completed(future_to_utt), total=len(load_tasks), 
                                    desc=f"Turn {tid} Loading", leave=False):
                result = future.result()
                if result is not None:
                    utt, data = result
                    wavinfo_dict[utt] = data
                # 如果 result 是 None，说明加载失败，直接跳过
        
        if not wavinfo_dict:
            continue
            
        # ================= 原有逻辑：并行保存 =================
        prefix = file_type
        parquet_list = []
        utt_keys = list(wavinfo_dict.keys())
        
        # 构建保存任务
        save_tasks = []
        for i in range(0, len(utt_keys), num_utts_per_parquet):
            batch_utts = utt_keys[i : i + num_utts_per_parquet]
            # 计算全局索引以生成唯一文件名
            global_idx = (start_idx + i) // num_utts_per_parquet
            pfile = f"{prefix}_{mid_name}_{global_idx:05d}.parquet"
            parquet_file = os.path.join(parquet_dir, pfile)
            
            save_tasks.append((wavinfo_dict, batch_utts, parquet_file))
            final_parquet2utt[pfile] = batch_utts

        # 使用进程池并行保存
        pool = multiprocessing.Pool(processes=num_save_processes)
        # map_async 是非阻塞的，但我们需要等待它完成才能进入下一个 Turn（为了内存管理）
        pool.starmap(save_parquet, save_tasks)
        pool.close()
        pool.join()

    # ================= 生成映射文件 =================
    utt2parquet_file = os.path.join(parquet_dir, 'utt2parquet.list')
    with open(utt2parquet_file, 'w') as u2pf:
        for pak, utts in final_parquet2utt.items():
            for utt in utts:
                u2pf.write(utt + '|' + pak + '\n')
                


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