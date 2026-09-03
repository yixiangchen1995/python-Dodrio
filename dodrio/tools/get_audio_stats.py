'''
FilePath: /python-Dodrio/dodrio/tools/get_audio_stats.py
Descripttion: 统计 package_dir 下所有 .pack 音频文件的总时长。
             package 中音频统一保存为 int16、单通道、48000 Hz。
Author: Yixiang Chen
version: 
Date: 2026-08-31
LastEditors: Yixiang Chen
LastEditTime: 2026-08-31
'''

import os
import glob
import argparse
from tqdm import tqdm

from dodrio.tools.load_data import load_data_from_line


def calc_pack_duration(pack_file, sample_rate=48000, bytes_per_sample=2, num_channels=1):
    '''
    根据 .pack 文件的字节大小计算音频时长（秒）。
    package 中音频格式固定为 int16 单通道。
    '''
    file_size = os.path.getsize(pack_file)
    num_samples = file_size / bytes_per_sample / num_channels
    duration = num_samples / sample_rate
    return duration


def format_duration(seconds):
    '''将秒数转换为 时:分:秒 格式'''
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_uttinfo_audio(package_dir, n):
    '''
    获取 uttinfo.list 中第 n 条（从 0 开始计数）语音数据。

    param {str} package_dir: package 目录路径
    param {int} n: uttinfo.list 中的索引，从 0 开始
    return {dict}: 包含 utt、pack_file、start、end、audio 的字典
    '''
    info_file = os.path.join(package_dir, 'uttinfo.list')
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"{info_file} 不存在")

    with open(info_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if n < 0 or n >= len(lines):
        raise IndexError(f"索引 {n} 超出范围，uttinfo.list 共有 {len(lines)} 条记录")

    utt, pf, start, end = lines[n].split('|')
    start = int(start)
    end = int(end)
    pack_file = os.path.join(package_dir, pf)
    # uttinfo.list 仅包含音频位置信息，构造与 load_data_from_line 兼容的最小行
    infoline = f"{utt}|{pack_file}|{start}|{end}|||"
    data_dict = load_data_from_line(infoline)

    return {
        'utt': data_dict['uttid'],
        'pack_file': pf,
        'start': start,
        'end': end,
        'audio': data_dict['audio'],
    }


def get_audio_stats(package_dir, sample_rate=48000):
    '''
    统计 package_dir 下所有 .pack 文件的音频总长度。
    '''
    pack_files = glob.glob(os.path.join(package_dir, '*.pack'))
    pack_files.sort()

    if not pack_files:
        print(f"在 {package_dir} 下未找到 .pack 文件")
        return

    total_duration = 0.0
    file_stats = []

    for pack_file in tqdm(pack_files, desc='统计音频时长'):
        duration = calc_pack_duration(pack_file, sample_rate=sample_rate)
        total_duration += duration
        file_stats.append((os.path.basename(pack_file), duration))

    print(f"\npackage_dir: {package_dir}")
    print(f"采样率: {sample_rate} Hz, 通道数: 1, 采样位宽: 16 bit")
    print(f"pack 文件数量: {len(pack_files)}")
    print(f"音频总长度: {format_duration(total_duration)} ({total_duration:.2f} 秒)")

    print("\n各 pack 文件时长:")
    for basename, duration in file_stats:
        print(f"  {basename}: {format_duration(duration)} ({duration:.2f} 秒)")


def main():
    parser = argparse.ArgumentParser(description='统计 package_dir 下所有 .pack 音频文件的总长度')
    parser.add_argument('package_dir', type=str, help='package 目录路径')
    parser.add_argument('--sample_rate', type=int, default=48000, help='统一采样率（默认 48000 Hz）')
    parser.add_argument('--get_utt', type=int, default=None, help='获取 uttinfo.list 中第 n 条（从 0 开始）语音数据并打印信息')
    args = parser.parse_args()

    if not os.path.isdir(args.package_dir):
        print(f"错误: {args.package_dir} 不是有效目录")
        return

    if args.get_utt is not None:
        try:
            info = get_uttinfo_audio(args.package_dir, args.get_utt)
            print(f"\n第 {args.get_utt} 条语音信息:")
            print(f"  utt: {info['utt']}")
            print(f"  pack_file: {info['pack_file']}")
            print(f"  start: {info['start']}")
            print(f"  end: {info['end']}")
            print(f"  audio shape: {info['audio'].shape}")
            print(f"  audio dtype: {info['audio'].dtype}")
            print(f"  audio duration: {len(info['audio']) / args.sample_rate:.2f} 秒")
        except Exception as e:
            print(f"获取第 {args.get_utt} 条语音失败: {e}")
        return

    get_audio_stats(args.package_dir, sample_rate=args.sample_rate)


if __name__ == '__main__':
    main()
