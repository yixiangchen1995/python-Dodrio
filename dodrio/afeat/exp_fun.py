'''
FilePath: /python-Dodrio/dodrio/afeat/exp_fun.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-27 09:59:30
LastEditors: Yixiang Chen
LastEditTime: 2025-04-29 17:33:45
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

