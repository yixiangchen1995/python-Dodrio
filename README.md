# Dodrio
Data format designed for TTS training

### 数据准备
首先需要准备含有wav或mp3格式音频的文件夹 

```python
test_data_dir = '/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/origin_data/test_data'
```

确定输出文件夹路径

```python
import os
outdir = '/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/testout'
dataname = 'test'
stockdir = outdir + '/stockdir'
usagedir = outdir + '/usagedir'
parquet_dir = os.path.join(stockdir, dataname, 'parquet_dir')
pack_dir = os.path.join(usagedir, dataname, 'pack_dir')

```

parquet_dir 为 parquet 格式文件夹
pack_dir 为package格式文件夹
info_outdir 为text和spk等信息的存储文件夹

### 数据打包

```python
import dodrio

# 输入 test_data_dir 生成 parquet 数据包
dodrio.gen_parquet(test_data_dir, parquet_dir, mid_name=dataname, file_type='wav')

# 输入 test_data_dir 生成 package 数据包 注意需要指定 采样率，pack会统一音频采样率存储
dodrio.gen_package(test_data_dir, pack_dir, mid_name=dataname, target_sample_rate=48000, file_type='wav')

# 也可以通过parquet数据格式生成package数据包
dodrio.parquet2package(parquet_dir, pack_dir, sample_rate=48000)
```

### 还原音频
将数据包中数据还原成音频

```python
reout = '/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/reout'
re_paruquet = os.path.join(reout, 'reparquet_dir')
re_pack = os.path.join(reout, 'repack_dir')

# parquet还原音频是还原成原始的格式，比如之前是mp3还原后还是mp3，且采样率这些不变
dodrio.parquet2wav(parquet_dir, re_paruquet)

# package 还原音频只会还原成 特定采样率的wav，对应采样率在一开始打包的时候已经设定好，且比特率和通道这些都固定
dodrio.package2wav(pack_dir, re_pack)
```


### 打包文本信息
打包存储 text 等信息

```python

# parquet
info_type = 'libritts'
info_outdir = os.path.join(stockdir, dataname, 'info_dir')
dodrio.gen_infodir(parquet_dir, test_data_dir, info_outdir, info_type, kl=['text', 'unnorm_text'], lang='en', from_type='parquet')

# package
info_type = 'libritts'
pack_info_outdir = os.path.join(usagedir, dataname, 'info_dir')
dodrio.gen_infodir(pack_dir, test_data_dir, pack_info_outdir, info_type, kl=['text', 'unnorm_text'], lang='en', from_type='pack')
```

parquet 和 package 调用的函数相同，都是 gen_infodir 。

需要注意的是 因为原始数据的存储方式千奇百怪，且文本不会按照唯一方式存储，所以调用的访问函数实际是不同的，这里预设了几种数据类型，比如上面的代码中就是从 libritts里加载数据格式

函数中的参数 第一个 parquet_dir 为 打包好的音频数据文件夹，这里主要是为了和打包数据分块列表一致所以载入；第二个参数 test_data_dir为文本等信息存储的文件夹；第三个参数 info_outdir 为 info的输出文件夹。

参数 info_type 为指定数据类型，目前只支持几种特定数据类型。 参数 kl 是keys list 这是因为有时文本有不同版本的文本，所以在这里设定一个帮助参数， 参数 lang 为language的默认值，数据文件有时会不带语种标签，在这里可以硬指定。

### 特征提取

可以用 extract_feat 提取特征并存储

```python
dodrio.extract_feat(extractor_func, featname, input_dir, out_dir, from_type, **params)
```

extractor_func 为 特征提取函数， featname 为对应特征名， input_dir为对应 package数据包，out_dir为输出文件夹， from_type为输入的数据包类型（目前仅支持 package）；params为特征提取所需额外参数

以cosyvoice的embedding举例，目前内置了对应的特征提取函数

```python

# 预设模型加载
from dodrio.afeat.exp_fun import extractor_embedding
tt = extractor_embedding(onnx_path)
extractor_func = tt.extractor

from_type = 'package'
featname = 'embed'
input_dir = pack_dir
out_dir = os.path.join(usagedir, dataname, featname+'_dir')

# 准备需要的额外参数 utt2spk
utt2spk = dodrio.get_utt2spk(info_outdir)

# 提取 embedding
dodrio.extract_feat(extractor_func, featname, input_dir, out_dir, from_type, utt2spk=utt2spk)

# 根据 spk 计算 spk embedding 均值
tt.mean_spk_embedding()

# 提取 spk 的平均 embedding
featname = 'spkembed'
input_dir = pack_dir
out_dir = os.path.join(usagedir, dataname, featname+'_dir')
extractor_func = tt.spk_embedding_save
dodrio.extract_feat(extractor_func, featname, input_dir, out_dir, from_type, utt2spk=utt2spk)
```

### 准备训练所需列表

目前也有预设的列表准备版本

```python
# supdir_list 可以包含多个数据包 目录
supdir_list = [os.path.join(usagedir, dataname)]
listoutdir= 'listoutdir'
# featlist 为需要添加的特征
featlist= ['embed', 'spkembed', 'speechtoken']

# check_func 为数据筛选函数， prefix 为 数据表名前缀
dodrio.gen_datalist(supdir_list, listoutdir, featlist, dodrio.check_func, prefix='test')
```

### 数据读取

以上面表格为例，加载单条数据可以通过 load_data_from_line 得到

```python
infoline = '296_142727_000010_000000|/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/testout/usagedir/test/pack_dir/wav_test_00000.pack|119946232|121158716|4|This reduction, if admitted, would much facilitate the introduction of emotion into our system, which, being founded on the distinction between the consciousness and the object, is likewise an intellectualist system.|en|embed|/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/testout/usagedir/test/embed_dir/wav_test_00000.embed|135168|135936|192|spkembed|/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/testout/usagedir/test/spkembed_dir/wav_test_00000.spkembed|135168|135936|192|speechtoken|/home/jovyan/chenyixiang/workspace/20250324_dodrio/testdata/testout/usagedir/test/speechtoken_dir/wav_test_00000.speechtoken|250040|252568|632'

data_dict = dodrio.load_data_from_line(infoline)
# data_dict.keys()
# dict_keys(['uttid', 'audio', 'spkid', 'text', 'language', 'embed', 'spkembed', 'speechtoken'])

```