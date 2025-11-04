'''
FilePath: /python-Dodrio/dodrio/afeat/exp_fun.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-27 09:59:30
LastEditors: Yixiang Chen
LastEditTime: 2025-11-04 10:44:51
'''

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

import logging

import whisper

class extractor_embedding:
    def __init__(self, onnx_path) -> None:
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
        
        self.spk2embedding = {}
    
    def extractor(self, floatnpwav, utt, utt2spk, sample_rate=48000):
        audio = torch.from_numpy(floatnpwav)
        audio = audio.float().unsqueeze(0)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[-1] < 400:
            print(utt)
            audio = torch.nn.functional.pad(audio, (0,400 - audio.shape[-1]), value=0)
        feat = kaldi.fbank(audio,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        if utt in utt2spk.keys():
            spk = utt2spk[utt]
        else:
            spk = 'SpkNone'
        if spk not in self.spk2embedding:
            self.spk2embedding[spk] = []
        self.spk2embedding[spk].append(embedding)
        embedding = np.array(embedding)
        return embedding
    
    def mean_spk_embedding(self):
        for k, v in self.spk2embedding.items():
            self.spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    
    def spk_embedding_save(self, floatnpwav, utt, utt2spk):
        if utt in utt2spk.keys():
            spk = utt2spk[utt]
        else:
            spk = 'SpkNone'
        spkembedding = self.spk2embedding[spk]
        spkembedding = np.array(spkembedding)
        return spkembedding

class speech_token_extractor:
    def __init__(self, onnx_path):
        #onnxruntime-gpu==1.16.0; sys_platform == 'linux'
        #onnxruntime==1.16.0; sys_platform == 'darwin' or sys_platform == 'windows'

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
    
    def extractor(self, floatnpwav, utt, sample_rate=48000):
        audio = torch.from_numpy(floatnpwav)
        audio = audio.float().unsqueeze(0)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[1] / 16000 > 30:
            logging.warning('do not support extract speech token for audio longer than 30s')
            speech_token = None
        else:
            feat = whisper.log_mel_spectrogram(audio, n_mels=128)
            speech_token = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                  self.ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
            speech_token = np.array(speech_token)
        return speech_token


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import librosa

import os
import sys
class suppress_stdout_stderr(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, *_):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class emotion2vec_extractor:
    def __init__(self):
        self.inference_pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model="iic/emotion2vec_base_finetuned",
            disable_update=True)
    
    def extractor(self, floatnpwav, utt, sample_rate=48000):
        floatnpwav = librosa.resample(floatnpwav, orig_sr=sample_rate, target_sr=16000)
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        with suppress_stdout_stderr():
            rec_result = self.inference_pipeline(input=floatnpwav.astype(np.float32), granularity="utterance", extract_embedding=True)
        scores = rec_result[0]['scores']
        embedding = rec_result[0]['feats']
        str_scores = [str(xx) for xx in scores]
        str_scores = ','.join(str_scores)
        return embedding, [str_scores]


from transformers import BertTokenizer, BertModel

class BertInfer():
    def __init__(self, cache_dir, device='cpu'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir=cache_dir)
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased", cache_dir=cache_dir)
        self.device = device
        if device == 'cuda':
            self.model = self.model.cuda() 

    def extractor(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        if self.device == 'cuda':
            for i in encoded_input:
                encoded_input[i] = encoded_input[i].cuda()
        output = self.model(**encoded_input, output_hidden_states=True)
        hid_feat = torch.cat(output["hidden_states"][-3:-2], -1)[0]
        hid_feat = hid_feat.detach().cpu().numpy()
        return hid_feat, ['0']

from core.package_base import package2wav
from tools.load_data import load_textinfo_data

import re
from pypinyin import lazy_pinyin, Style
def label_pre(par_str):
    tmp = []
    for ele in par_str.split(']'):
        if ele == '':
            continue
        elif ele[0] == '[':
            tmp.append(ele+']')
        else:
            tmp.append(ele)
    return tmp
def pypre(pylist):
    outlist = []
    for idx, ele in enumerate(pylist):
        if ele == '':
            continue
        elif ele[0] == '[':
            outlist.extend(label_pre(ele))
        else:
            outlist.append(ele)
    return outlist

def rm_punc(text):
    rr = '。？！，,.?!'
    outt = re.sub(r"[%s]+" % rr, "", text)
    return outt

def gen_mfadir(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    package_dir = os.path.join(datadir, 'pack_dir')
    package2wav(package_dir, outdir) 
    infodir = os.path.join(datadir, 'info_dir') 
    infofile = os.path.join(infodir, 'uttinfo_text.list')
    info_dict = load_textinfo_data(infofile)
    for basename in info_dict.keys():
        text = info_dict[basename]['text'] 
        usetext = rm_punc(text)
        pylist = lazy_pinyin(usetext, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True)
        pylist = pypre(pylist)
        tfile = os.path.join(outdir, basename+'_text.txt')
        pyfile = os.path.join(outdir, basename+'.txt')
        with open(tfile, 'w') as tfw:
            tfw.write(text)
        with open(pyfile, 'w') as pyfw:
            pyfw.write(' '.join(pylist))



