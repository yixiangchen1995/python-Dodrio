import os
import pandas as pd

import dodrio

def process_data_from_dir(info_dir, kl=['text']):
    """
    从指定目录下读取 all_paths.txt，处理其中列出的每个文件夹下的 result_step2.csv
    
    Args:
        root_dir (str): 包含 all_paths.txt 的根目录路径
                        例如: /home/deepspeed/workdir/20260622_data_process/oridatadirlist_save/test2dir
                        
    Returns:
        tuple: (info_dict, keys_list)
            - info_dict: dict, key为id, value为该行的完整数据字典
            - keys_list: list, 字典中key的顺序列表，前4项固定为 ['id', 'speaker', 'text', 'language']
    """
    
    # 1. 构建 all_paths.txt 的全路径
    input_txt_path = os.path.join(info_dir, "all_paths.txt")
    
    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"File not found: {input_txt_path}")
        
    # 2. 读取文件夹列表
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        # 去除空行和首尾空白
        folder_list = [line.strip() for line in f if line.strip()]

    info_dict = {}
    keys_list = []
    is_first_entry = True # 标记是否是第一个有效处理的CSV，用于初始化 keys_list

    # 定义固定的前4个键
    fixed_keys_prefix = ['id', 'speaker', 'text', 'language']

    print(f"Start processing {len(folder_list)} folders...")

    for i, folder_path in enumerate(folder_list):
        csv_path = os.path.join(folder_path, 'result_step2.csv')
        
        # 进度打印（可选）
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(folder_list)} folders...")

        if not os.path.exists(csv_path):
            # 如果某个文件夹下没有csv，跳过并警告
            # print(f"Warning: CSV file not found in {folder_path}, skipping.")
            continue
            
        try:
            # 使用 pandas 读取 CSV
            # dtype=str 确保所有数据作为字符串处理，避免数字被自动转换导致格式问题
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        if df.empty:
            continue

        # --- 初始化 keys_list (仅执行一次) ---
        if is_first_entry:
            original_columns = df.columns.tolist()
            # 策略：固定前缀 + 原始CSV中除了前缀名以外的所有列
            # 注意：如果原始CSV中有 'id' 列，它会被我们的逻辑覆盖，但在keys_list中我们通常只展示一次
            # 这里为了保持顺序：先放固定的4个，再放剩下的原始列
            other_keys = [col for col in original_columns if col not in fixed_keys_prefix]
            keys_list = fixed_keys_prefix + other_keys
            is_first_entry = False

        # --- 遍历每一行数据 ---
        for index, row in df.iterrows():
            # 1. 获取 denoised_file 原始值
            denoised_file_raw = row.get('denoised_file', None)
            
            # 如果 denoised_file 为空或 NaN，跳过该行
            if pd.isna(denoised_file_raw) or not denoised_file_raw:
                continue
            
            # 2. 处理 ID (audio_id)
            # 去除路径，只留文件名
            filename_with_ext = os.path.basename(denoised_file_raw)
            # 去除 .wav 后缀
            if filename_with_ext.lower().endswith('.wav'):
                audio_id = filename_with_ext[:-4]
            else:
                audio_id = filename_with_ext 
            
            # 3. 处理 Speaker
            pred_spk = row.get('pred_spk', '')
            if pd.isna(pred_spk):
                pred_spk = ''
            
            # 获取根目录前缀
            # 将反斜杠替换为正斜杠以统一处理，然后分割
            path_parts = denoised_file_raw.replace('\\', '/').split('/')
            # 取第一部分作为根目录，例如 "101次抢婚_622480/01/..." -> "101次抢婚_622480"
            root_dir_prefix = path_parts[0] if len(path_parts) > 0 else ""
            
            speaker_val = f"{root_dir_prefix}__{pred_spk}"
            
            # 4. 处理 Text
            text_val = row.get('paralin', '')
            if pd.isna(text_val):
                text_val = ''
                
            # 5. 处理 Language
            language_val = "ZH"
            
            # 6. 构建当前行的完整字典
            # 先将整行转为字典，保留所有原始列
            row_data = row.to_dict()
            
            # 覆盖/设置特定的4个字段
            row_data['id'] = audio_id
            row_data['speaker'] = speaker_val
            row_data['text'] = text_val
            row_data['language'] = language_val
            
            # 7. 存入 info_dict
            # key 为计算出的 id
            final_id_key = row_data['id']
            
            # 如果存在重复ID，后面的会覆盖前面的。
            # 如果需要保留所有重复项，可以将 value 改为列表，但通常ID是唯一的
            info_dict[final_id_key] = row_data

    print(f"Processing complete. Total unique IDs: {len(info_dict)}")
    return info_dict, keys_list


test_data_dir = "/home/deepspeed/workdir/20260622_data_process/oridatadirlist_save/test2dir"
#test_data_dir = '/home/deepspeed/workdir/20260622_data_process/oridatadirlist_save/youku_tv_aa'
import os
outdir = '/home/deepspeed/model_output/dodrio_data/data_save'

#dataname = 'youkuTVSplitaa'
dataname = 'testdataset'
stockdir = outdir + '/stockdir'
usagedir = outdir + '/usagedir'
parquet_dir = os.path.join(stockdir, dataname, 'parquet_dir')
pack_dir = os.path.join(usagedir, dataname, 'pack_dir')
info_outdir = os.path.join(usagedir, dataname, 'info_dir')

info_type = 'aaa'
pack_info_outdir = os.path.join(usagedir, dataname, 'info_dir')

dodrio.gen_infodir(pack_dir, test_data_dir, pack_info_outdir, info_type, kl=['text'], lang='zh', from_type='pack', info_func=process_data_from_dir)