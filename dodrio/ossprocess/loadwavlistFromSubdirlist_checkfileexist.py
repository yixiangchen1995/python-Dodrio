import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import time

warnings.filterwarnings('ignore')

def process_single_subdir_verified(args):
    """
    工作函数：
    1. 读取 CSV
    2. 构建绝对路径
    3. 【新增】验证文件是否存在，剔除无效数据
    """
    subdir_abs_path_str = args
    subdir_path = Path(subdir_abs_path_str)
    
    csv_file = subdir_path / "result_step2.csv"
    
    if not csv_file.is_file():
        return {}, []
    
    try:
        # 1. 高效读取
        df = pd.read_csv(
            csv_file, 
            usecols=['final_result', 'denoised_file'],
            dtype={'denoised_file': str},
            on_bad_lines='skip'
        )
        
        if df.empty:
            return {}, []
            
        # 2. 过滤 final_result == True
        mask = df['final_result'].astype(str).str.lower() == 'true'
        filtered_df = df.loc[mask, ['denoised_file']]
        
        if filtered_df.empty:
            return {}, []
            
        # 3. 清洗数据
        filtered_df['denoised_file'] = filtered_df['denoised_file'].str.strip().str.strip('"').str.strip("'")
        filtered_df = filtered_df.dropna(subset=['denoised_file'])
        filtered_df = filtered_df[filtered_df['denoised_file'] != '']
        
        if filtered_df.empty:
            return {}, []

        # 4. 构建绝对路径逻辑
        subdir_name = subdir_path.name
        prefix_to_remove = subdir_name + "/"
        prefix_to_remove_win = subdir_name + "\\"
        
        # 去除 CSV 路径中的子目录前缀
        relative_rest_path = filtered_df['denoised_file'].str.replace(prefix_to_remove, "", n=1, regex=False)
        relative_rest_path = relative_rest_path.str.replace(prefix_to_remove_win, "", n=1, regex=False)
        
        # 拼接完整绝对路径
        base_path = subdir_abs_path_str.replace("\\", "/")
        clean_rest = relative_rest_path.str.replace("\\", "/")
        full_paths_series = base_path + "/" + clean_rest
        
        # 提取 Key (文件名去后缀)
        file_names_with_ext = full_paths_series.str.split("/").str[-1]
        keys_series = file_names_with_ext.str[:-4] # 假设 .wav
        
        # --- 【核心修改】验证文件存在性 ---
        
        # 将 Series 转换为 Python 列表，以便快速迭代检查
        paths_list = full_paths_series.tolist()
        keys_list = keys_series.tolist()
        
        valid_keys = []
        valid_paths = []
        
        # 遍历当前批次的所有路径，检查是否存在
        # 注意：这里是在单个进程内循环，但由于每个进程的 CSV 行数通常不多（几百到几千行），
        # 所以这个循环非常快，不会成为全局瓶颈。
        for k, p in zip(keys_list, paths_list):
            if os.path.isfile(p): # 使用 isfile 比 exists 更快且更准确（排除目录）
                valid_keys.append(k)
                valid_paths.append(p)
                
        if not valid_keys:
            return {}, []
            
        # 构建局部结果
        local_dict = dict(zip(valid_keys, valid_paths))
        local_list = valid_keys
        
        return local_dict, local_list
        
    except Exception as e:
        # print(f"Error processing {subdir_abs_path_str}: {e}")
        return {}, []

def process_audio_directory_verified(input_dir: str, max_workers: int = None, test_count: int = None) -> tuple[dict, list]:
    """
    高性能且带文件存在性验证的处理函数。
    """
    #print(f"正在读取子目录列表: {list_file_path} ...")
    list_file_path = os.path.join(input_dir, "all_paths.txt")
    
    with open(list_file_path, 'r', encoding='utf-8') as f:
        subdir_list = [line.strip() for line in f if line.strip()]
        
    if not subdir_list:
        raise ValueError("子目录列表为空")
        
    if test_count is not None:
        subdir_list = subdir_list[:test_count]
        print(f"[测试模式] 仅处理前 {len(subdir_list)} 个目录")
    else:
        print(f"[全量模式] 共加载 {len(subdir_list)} 个子目录路径")
        
    tasks = subdir_list
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
        
    final_dict = {}
    final_list = []
    processed_count = 0
    total_files_found = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {executor.submit(process_single_subdir_verified, task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(future_to_info):
            try:
                res_dict, res_list = future.result()
                if res_dict:
                    final_dict.update(res_dict)
                    final_list.extend(res_list)
                    total_files_found += len(res_list)
                    
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"已扫描 {processed_count}/{len(tasks)} 个目录, 累计找到有效文件: {total_files_found}")
                    
            except Exception as e:
                idx = future_to_info[future]
                print(f"任务 {idx} 异常: {e}")
                
    end_time = time.time()
    print(f"\n✅ 处理完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终有效文件数: {len(final_list)}")
    
    return final_dict, final_list

if __name__ == "__main__":
    # ================= 配置区域 =================
    #LIST_FILE = "./subdirs_list.txt" 
    INPUT_DIR = "/home/deepspeed/workdir/20260622_data_process/oridatadirlist_save/test2dir" 
    WORKERS = 16 # 建议根据 CPU 核心数调整
    # ===========================================
    
    try:
        # --- 1. 小规模测试 ---
        print("="*30)
        print("开始测试模式 (前 5 个目录)...")
        test_dict, test_list = process_audio_directory_verified(
            input_dir=INPUT_DIR,
            max_workers=4,
            test_count=5
        )
        
        if test_list:
            print("\n--- 测试结果验证 ---")
            first_key = test_list[0]
            first_path = test_dict[first_key]
            print(f"Key: {first_key}")
            print(f"Path: {first_path}")
            print(f"len test_list: {len(test_list)}")
            # 双重验证
            if os.path.exists(first_path):
                print("✅ 路径存在，逻辑正确！")
            else:
                print("❌ 严重错误：代码认为文件存在，但实际检查不存在。")
        else:
            print("⚠️ 测试未找到任何有效文件。")

        # --- 2. 全量运行 (确认无误后取消注释) ---
        # print("\n" + "="*30)
        # print("开始全量运行...")
        # full_dict, full_list = process_audio_directory_verified(
        #     list_file_path=LIST_FILE,
        #     max_workers=WORKERS,
        #     test_count=None
        # )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
