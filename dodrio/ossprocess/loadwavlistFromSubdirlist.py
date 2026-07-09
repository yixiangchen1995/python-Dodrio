import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import time

warnings.filterwarnings('ignore')

def process_single_subdir_smart(args):
    """
    智能处理函数：
    1. 读取子目录绝对路径
    2. 读取 CSV
    3. 自动从 CSV 路径中去除子目录名前缀，拼接到子目录绝对路径后
    """
    subdir_abs_path_str = args
    subdir_path = Path(subdir_abs_path_str)
    
    csv_file = subdir_path / "result_step2.csv"
    
    if not csv_file.is_file():
        return {}, []
    
    try:
        df = pd.read_csv(
            csv_file, 
            usecols=['final_result', 'denoised_file'],
            dtype={'denoised_file': str},
            on_bad_lines='skip'
        )
        
        if df.empty:
            return {}, []
            
        # 过滤 True
        mask = df['final_result'].astype(str).str.lower() == 'true'
        filtered_df = df.loc[mask, ['denoised_file']]
        
        if filtered_df.empty:
            return {}, []
            
        # 清洗数据
        filtered_df['denoised_file'] = filtered_df['denoised_file'].str.strip().str.strip('"').str.strip("'")
        filtered_df = filtered_df.dropna(subset=['denoised_file'])
        filtered_df = filtered_df[filtered_df['denoised_file'] != '']
        
        if filtered_df.empty:
            return {}, []

        # --- 核心逻辑修改 ---
        
        # 获取当前子目录的名称 (例如: "101次抢婚_622480")
        subdir_name = subdir_path.name
        
        # CSV 中的路径通常是: "101次抢婚_622480/01/denoise/file.wav"
        # 我们需要去掉前面的 "101次抢婚_622480/"，剩下 "01/denoise/file.wav"
        
        # 构造要去除的前缀字符串 (注意加上斜杠)
        prefix_to_remove = subdir_name + "/"
        # 兼容 Windows 反斜杠
        prefix_to_remove_win = subdir_name + "\\"
        
        # 使用 str.replace 去除前缀 (只替换一次，且只在开头匹配更安全，但 replace 简单高效)
        # 更严谨的做法是用 str.startswith 判断后切片，但 replace 对于标准数据足够快
        relative_rest_path = filtered_df['denoised_file'].str.replace(prefix_to_remove, "", n=1, regex=False)
        relative_rest_path = relative_rest_path.str.replace(prefix_to_remove_win, "", n=1, regex=False)
        
        # 现在 relative_rest_path 是 "01/denoise/file.wav"
        
        # 构建最终绝对路径: 子目录绝对路径 + 剩余相对路径
        # subdir_abs_path_str 已经是绝对路径了
        base_path = subdir_abs_path_str.replace("\\", "/")
        clean_rest = relative_rest_path.str.replace("\\", "/")
        
        # 拼接: /abs/path/to/subdir + "/" + rest/of/path
        full_paths_series = base_path + "/" + clean_rest
        
        # 提取文件名 Key
        file_names_with_ext = full_paths_series.str.split("/").str[-1]
        keys_series = file_names_with_ext.str[:-4] # 去除 .wav
        
        local_dict = dict(zip(keys_series, full_paths_series))
        local_list = keys_series.tolist()
        
        return local_dict, local_list
        
    except Exception as e:
        # print(f"Error: {e}")
        return {}, []

def process_audio_directory_no_root(input_dir: str, max_workers: int = None, test_count: int = None) -> tuple[dict, list]:
    """
    无需 root_dir 的版本。
    直接利用 txt 中的子目录绝对路径和 CSV 内容智能拼接。
    """
    #print(f"正在读取子目录列表: {list_file_path} ...")
    list_file_path = os.path.join(input_dir, "all_paths.txt")
    
    with open(list_file_path, 'r', encoding='utf-8') as f:
        subdir_list = [line.strip() for line in f if line.strip()]
        
    if not subdir_list:
        raise ValueError("子目录列表为空")
        
    if test_count is not None:
        subdir_list = subdir_list[:test_count]
        print(f"[测试模式]仅处理前 {len(subdir_list)} 个目录")
    else:
        print(f"[全量模式]共加载 {len(subdir_list)} 个子目录路径")
        
    tasks = subdir_list # 直接传递子目录绝对路径字符串
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
        
    final_dict = {}
    final_list = []
    processed_count = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {executor.submit(process_single_subdir_smart, task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(future_to_info):
            try:
                res_dict, res_list = future.result()
                if res_dict:
                    final_dict.update(res_dict)
                    final_list.extend(res_list)
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"已处理 {processed_count}/{len(tasks)} ...")
            except Exception as e:
                idx = future_to_info[future]
                print(f"任务 {idx} 失败: {e}")
                
    end_time = time.time()
    print(f"\n处理完成！耗时: {end_time - start_time:.2f} 秒, 有效文件: {len(final_list)}")
    
    return final_dict, final_list

if __name__ == "__main__":
    INPUT_DIR = "/home/deepspeed/workdir/20260622_data_process/oridatadirlist_save/test2dir" 
    
    try:
        # 测试模式
        test_dict, test_list = process_audio_directory_no_root(
            input_dir=INPUT_DIR,
            max_workers=16,
            test_count=None
        )
        
        if test_list:
            print("\n--- 验证 ---")
            k = test_list[0]
            v = test_dict[k]
            print(f"Key: {k}")
            print(f"Path: {v}")
            print(f"len test_list: {len(test_list)}")
            if os.path.exists(v):
                print("✅ 成功！无需 root_dir 也能正确定位。")
            else:
                print("❌ 文件不存在，请检查路径逻辑。")
                
    except Exception as e:
        import traceback
        traceback.print_exc()
