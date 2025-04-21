'''
FilePath: /python-Dodrio/dodrio/afeat/exp_audiof.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-04-09 14:56:10
LastEditors: Yixiang Chen
LastEditTime: 2025-04-10 17:59:03
'''

from librosa.filters import mel as librosa_mel_fn
import torch 
import torch.nn.functional as F
import numpy as np
from functools import partial

# for pitch
import pyworld as pw

mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_extractor_torch(wav, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''
        wav shape : (B, sample_len) torch.Tensor, range [-1, 1], Float 32
        spec out shape : (B, mel_dim, spec_len) torch.Tensor, Float 32
        Float 32  because mel_basis by librosa
    '''
    if torch.min(wav) < -1.0:
        print("min value is ", torch.min(wav))
    if torch.max(wav) > 1.0:
        print("max value is ", torch.max(wav))

    # Get Mel Wrap Matix
    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(wav.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(wav.device)] = torch.from_numpy(mel).float().to(wav.device)
        hann_window[str(wav.device)] = torch.hann_window(win_size).to(wav.device)

    if wav.shape[-1] > int((n_fft - hop_size) / 2):
        wav_pad = torch.nn.functional.pad(
            wav.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
        )
    else:
        wav_pad = torch.nn.functional.pad(
            wav.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='constant'
        )

    wav_pad = wav_pad.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            wav_pad,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(wav_pad.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    ) #  [B, nfft//2+1, seq_len, 2]

    #phase = torch.atan2(spec[...,1], spec[...,0])

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    magnitudes = spec # [B, nfft//2+1, seq_len]

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(wav_pad.device)], spec)
    # default 1e-5 clip norm
    spec = spectral_normalize_torch(spec)

    return spec, magnitudes


def pitch_extract_pyworld(wav, mel_len, sample_rate, hop_size, etype='harvest'):
    '''
        wav shape : [1, sample_len] torch.Tensor, range [-1, 1]
        out f0 shape : [seq_len] torch.Tensor 
    '''
    frame_period = hop_size * 1000 / sample_rate
    if etype == 'harvest':
        _f0, t = pw.harvest(wav.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
    else:
        _f0, t = pw.dio(wav.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
    if sum(_f0 != 0) < 5: # this happens when the algorithm fails
        _f0, t = pw.dio(wav.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period) # if harvest fails, try dio
    f0 = pw.stonemask(wav.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
    if mel_len > 1:
        f0 = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=mel_len, mode='linear').view(-1)
    return f0

def energy_extract_time(wav, hop_size, win_size):
    # wav shape : [1, sample_len] torch.Tensor, range [-1, 1]
    # out shape : [seq_len] torch.Tensor 
    feat_len = wav.shape[-1] // hop_size
    energy_list = []
    for idx in range(feat_len):
        piece = wav[:,idx*hop_size:(idx+1)*hop_size]
        energy = torch.sum(piece**2)
        energy_list.append(energy)
    return torch.tensor(energy_list)

def energy_extract_spec(magnitudes):
    # magnitudes [B, nfft//2+1, seq_len]
    # magnitudes 2-norm on seq_len dim 
    energy = torch.norm(magnitudes, dim=1)
    return energy
        


pitch_fun_dict = {
    'pyworld': pitch_extract_pyworld,
}
mel_fun_dict = {
    'torch': mel_extractor_torch,
}

def extract_mfe(floatnpwav, utt, meltype='torch', f0type='pyworld', energytype='spec',
                sampling_rate=48000, hop_size=512, 
                n_fft=2048, num_mels=80, win_size=2048, fmin=0, fmax=8000, mel_center=False,
                f0etype='harvest',
                ):
    '''
        wav shape : (B, sample_len) torch.Tensor, range [-1, 1], Float 32
        f0 shape : [seq_len] torch.Tensor 
        energy shape : [seq_len] torch.Tensor 
        mel_spec out shape : (B, mel_dim, spec_len) torch.Tensor, Float 32
    '''

    wav = torch.from_numpy(floatnpwav)
    wav = wav.float().unsqueeze(0)
    if wav.shape[-1] < n_fft:
        wav = torch.nn.functional.pad(
            wav, (n_fft//2 +1, n_fft//2+1), mode='constant'
        )

    pitch_extractor = partial(pitch_fun_dict[f0type], 
                                sample_rate=sampling_rate, hop_size=hop_size,
                                etype=f0etype)
    mel_extractor = partial(mel_fun_dict[meltype],
                                n_fft=n_fft, num_mels=num_mels, 
                                sampling_rate=sampling_rate, hop_size=hop_size, win_size=win_size, 
                                fmin=fmin, fmax=fmax, center=mel_center) 
    
    mel_spec, magnitudes = mel_extractor(wav) 
    pitch = pitch_extractor(wav, mel_spec.shape[-1]) 

    if energytype == 'time':
        energy_extractor = partial(energy_extract_time, 
                                    hop_size=hop_size, win_size=win_size)
        energy = energy_extractor(wav)
        if mel_spec.shape[-1]-energy.shape[-1] > 0:
            energy = F.pad(energy, (0, mel_spec.shape[-1]-energy.shape[-1]))
        energy = energy[...,:mel_spec.shape[-1]]
    else:
        energy_extractor = energy_extract_spec
        energy = energy_extractor(magnitudes)
    
    pitch = pitch.squeeze(0).cpu().numpy().astype(np.float32)
    energy = energy.squeeze(0).cpu().numpy().astype(np.float32)
    mel_spec = mel_spec.squeeze(0).cpu().numpy().astype(np.float32)

    return pitch, energy, mel_spec



