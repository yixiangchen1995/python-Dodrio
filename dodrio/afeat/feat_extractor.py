'''
FilePath: /python-Dodrio/dodrio/afeat/feat_extractor.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-26 19:12:18
LastEditors: Yixiang Chen
LastEditTime: 2026-05-11 12:00:30
'''


import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from dodrio.tools.load_data import load_data_dict, load_pack_audio_data

from dodrio.afeat.exp_load import align_jsondict, align_jsondict_aa, align_jsondict_exc


import pyarrow.parquet as pq

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


from concurrent.futures import ThreadPoolExecutor
from queue import Queue

def extract_feat_tpool(extractor_func, featname, input_dir, out_dir, from_type, io_workers=4, queue_maxsize=5, **params):
    """
    :param io_workers: 用于加载数据的并发线程数。建议设置为 4-8，取决于磁盘IO性能。
    :param queue_maxsize: 队列最大长度。控制预加载的数据量，防止内存溢出。
                          如果每个pack很大，设小一点(2-3)；如果很小，可以设大一点(10+)。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if from_type == 'parquet':
        suffix = '.parquet'
        infolistf = ''
    else:
        suffix = '.pack'
        infolistf = os.path.join(input_dir, 'uttinfo.list')
        
    packlist = get_file_list(input_dir, suffix)
    packlist.sort()
    
    feat_info_file = os.path.join(out_dir, 'feat_info_' + featname + '.list')
    
    # 创建线程安全队列
    # maxsize 限制内存中同时存在的“待处理”数据包数量
    data_queue = Queue(maxsize=queue_maxsize)
    
    # 标记任务结束的信号
    SENTINEL = None

    # --- 1. 定义生产者函数 (仅负责加载数据) ---
    def load_worker(packfile):
        try:
            packid = os.path.split(packfile)[-1].split(suffix)[0]
            # 这里执行耗时的 I/O 操作
            uttdict = load_pack_audio_data(packfile, infolistf, return_sr=False)
            
            # 将加载好的数据放入队列
            # 如果队列满了，这里会阻塞，直到消费者取走数据，从而自动控制内存使用
            data_queue.put((packid, uttdict))
        except Exception as e:
            print(f"Error loading {packfile}: {e}")
            # 即使出错也放一个 None 或者跳过，这里简单处理为跳过，实际需根据业务逻辑调整
            # 为了保持计数对齐，最好还是放入某种错误标记，这里简化处理
            pass

    # --- 2. 启动生产者线程池 ---
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        # 提交所有加载任务
        # 注意：submit 是非阻塞的，它会迅速将所有任务扔给线程池
        futures = [executor.submit(load_worker, pf) for pf in packlist]
        
        # --- 3. 消费者逻辑 (主线程执行 GPU 计算和写入) ---
        # 主线程现在变成了消费者，从队列取数据，然后做 GPU 计算
        
        with open(feat_info_file, 'w') as oinfo_out:
            total_packs = len(packlist)
            processed_count = 0
            
            # 使用 tqdm 监控整体进度
            pbar = tqdm(total=total_packs, desc="GPU Processing & Writing")
            
            while processed_count < total_packs:
                # 从队列获取数据。如果队列为空，这里会阻塞等待生产者放入数据
                item = data_queue.get()
                
                if item is None:
                    # 如果收到停止信号（可选逻辑，这里主要靠计数控制）
                    continue
                
                packid, uttdict = item
                
                out_feat_path = os.path.join(out_dir, packid + '.' + featname)
                position = 0

                pbar_inner = tqdm(total=len(uttdict), desc=f"Pack: {packid}", position=1, leave=False, ncols=100)
                
                # GPU 计算和文件写入部分 (串行执行，保证顺序和显存管理简单)
                try:
                    with open(out_feat_path, 'wb') as outf:
                        for utt in uttdict.keys(): 
                            wavdata = uttdict[utt]
                            
                            # 【关键点】这里调用 GPU 计算
                            feat = extractor_func(wavdata, utt, **params)
                            
                            if feat is None:
                                feat = np.array([0]).astype(np.float32)
                            
                            feat = feat.astype(np.float32)
                            fshape = feat.shape
                            feat_flat = np.reshape(feat, -1)
                            
                            byte_feat = bytes(feat_flat)
                            outf.write(byte_feat)
                            
                            byte_num = len(feat_flat) * 4 
                            end_position = position + byte_num 
                            
                            feat_info_str = '|'.join([
                                utt, 
                                os.path.split(out_feat_path)[-1], 
                                str(position), 
                                str(end_position), 
                                ','.join([str(xx) for xx in fshape])
                            ])
                            oinfo_out.write(feat_info_str + '\n')
                            
                            position += byte_num

                            pbar_inner.update(1)

                except Exception as e:
                    print(f"Error processing features for {packid}: {e}")
                finally:
                    pbar_inner.close()
                    # 重要：手动删除大对象，帮助垃圾回收，防止内存累积
                    del uttdict
                    del item
                    
                processed_count += 1
                pbar.update(1)
                
        pbar.close()

    # 确保所有加载线程都已完成（虽然队列取完了通常意味着工作做完了，但显式等待更稳妥）
    for f in futures:
        f.result()
        
    #print("All tasks completed.")


def extract_feat_extrainfo(extractor_func, featname, input_dir, out_dir, from_type, **params):
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
            try:
                feat, extrainfo = extractor_func(wavdata, utt, **params)
            except:
                feat = None
                extrainfo = ['0']
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
            feat_info.extend(extrainfo)
            info_outline = '|'.join(feat_info) + '\n'
            oinfo_out.write(info_outline) 

            position += byte_num
        outf.close()
    oinfo_out.close()

def extract_bert_extrainfo(extractor_func, featname, input_dir, out_dir, from_type, **params):
    os.makedirs(out_dir, exist_ok=True)
    #uttinfo_list = os.path.join(input_dir, 'uttinfo_text.list')
    suffix = '.info' 
    packlist = get_file_list(input_dir, suffix)
    packlist.sort()

    feat_info_file = os.path.join(out_dir, 'feat_info_'+featname+'.list')
    oinfo_out = open(feat_info_file, 'w')

    for pidx in range(len(packlist)):
        info_parquet_file = packlist[pidx] 
        packid = os.path.split(info_parquet_file)[-1].split(suffix)[0]

        out_feat_path = os.path.join(out_dir, packid+'.'+featname)
        outf = open(out_feat_path, 'wb')
        position = 0

        df = pq.read_table(info_parquet_file).to_pandas()
        #num_utts_per_parquet = len(df)
        for idx in tqdm(range(len(df)), desc=f'{packid} Processing'):
            if df.iloc[idx]['id'] == None:
                continue
            utt = df.iloc[idx]['id']
            text = df.iloc[idx]['text'] 

            try:
                feat, extrainfo = extractor_func(text, **params)
            except:
                feat = None
                extrainfo = ['0']
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
            feat_info.extend(extrainfo)
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


