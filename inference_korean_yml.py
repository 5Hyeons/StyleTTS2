import parser
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import argparse

from models import *
from utils import *
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


PAD = '_'
BOS = '<bos>'
EOS = '<eos>'
PUNC = '!?\'\"().,-=:;^&*~'
SPACE = ' '
_SILENCES = ['sp', 'spn', 'sil']

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
symbols = [PAD] + [BOS] + [EOS] + list(VALID_CHARS) + _SILENCES

id_to_sym = {i: sym for i, sym in enumerate(symbols)}
#---
dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

from g2pK.g2pkc import G2p

g2pk = G2p()
class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text, cleaned=False):
        indexes = []
        if not cleaned:
            text = g2pk(text)
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

textclenaer = TextCleaner()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, model):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def get_model(config, ckpt_path):

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, None, None, None)

    params_whole = torch.load(ckpt_path, map_location='cpu')
    params = params_whole['net']

    ignore_modules = ['bert', 'bert_encoder', 'text_aligner', 'pitch_extractor', 'mpd', 'msd', 'wd']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                print(f'{key} key lenghth: {len(model[key].state_dict().keys())}, state_dict length: {len(state_dict.keys())}')
                for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
                model[key].eval()
                model[key].to(device)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    return model, model_params, sampler

def inference(model, model_params, sampler, tokens, s_prev, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    t = 0.7
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        d_en = model.prosodic_text_encoder(tokens, input_lengths, text_mask)
        d_en_dur = d_en.transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=d_en_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new
        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()[..., :-50], s_prev # weird pulse at the end of the model, need to be fixed later

def evaludation(model, model_params, sampler, ref, root_path, lines, output_path):
    wavs = []
    silence = np.zeros(int(24000 * 0.5)) # 0.5 sec silence for interval
    # 
    ref_wav_path, alpha, beta, diffusion_steps, embedding_scale = ref.split('|')
    ref_s = compute_style(ref_wav_path, model)
    alpha = float(alpha)
    beta = float(beta)
    diffusion_steps = int(diffusion_steps)
    embedding_scale = float(embedding_scale)

    for i, line in enumerate(lines):
        line = line.strip()
        wav_orig_filename, text, sid = line.split('|')
        wav_orig_filepath = os.path.join(root_path, wav_orig_filename)
        wav_orig = librosa.load(wav_orig_filepath, sr=24000)[0]
        ref_output_path = output_path.replace('.wav', f'_ref_{i:02d}.wav')
        os.makedirs(os.path.dirname(ref_output_path), exist_ok=True)

        sf.write(ref_output_path, wav_orig, 24000, format='WAV', subtype='PCM_16')

        tokens = textclenaer(text)
        tokens.insert(0, 0)
        tokens.append(0)

        wav_synth, _ = inference(model, model_params, sampler, tokens, None, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        new_output_path = output_path.replace('.wav', f'_{i:02d}.wav')
        sf.write(new_output_path, wav_synth, 24000, format='WAV', subtype='PCM_16')
        
        if random.random() < 0.5:
            wavs.append(wav_orig)
            wavs.append(silence)
            wavs.append(wav_synth)
            wavs.append(silence)
            print(f"{text}: original first")
        else:
            wavs.append(wav_synth)
            wavs.append(silence)
            wavs.append(wav_orig)
            wavs.append(silence)
            print(f"{text}: fake first")

    audio = np.concatenate(wavs[:-1], axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, 24000, format='WAV', subtype='PCM_16')

def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    models = []
    for model in config['models']:
        model_path, model_config_path = model.split('|')
        model_path = 'Models/' + model_path
        model_config_path = 'Configs/' + model_config_path
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        model, model_params, sampler = get_model(model_config, model_path)
        models.append((model, model_params, sampler))

        if config['validation']['use']:
            output_path = os.path.join('Outputs', model_path.split('/')[1], 'evaluation', config['output_path'])
            with open(config['validation']['filelist_path'], 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines = lines[:config['validation']['num_lines']]
            root_path = model_config['data_params']['root_path']
            ref = config['validation']['reference']
            evaludation(model, model_params, sampler, ref, root_path, lines, output_path)


    texts = []
    for line in config['texts']:
        line = line.strip()
        tokens = textclenaer(line)
        tokens.insert(0, 0)
        tokens.append(0)
        texts.append(tokens)
    
    refs = []
    ref_wavs = []
    silence = np.zeros(int(24000 * 0.5)) # 0.5 sec silence for interval
    for line in config['references']:
        ref_wav_path, alpha, beta, diffusion_steps, embedding_scale = line.split('|')
        alpha = float(alpha)
        beta = float(beta)
        diffusion_steps = int(diffusion_steps)
        embedding_scale = float(embedding_scale)
        ref_s = compute_style(ref_wav_path, models[0][0])
        refs.append((ref_wav_path, ref_s, alpha, beta, diffusion_steps, embedding_scale)) 
        #
        ref_wav = librosa.load(ref_wav_path, sr=24000)[0]
        ref_wavs.append(ref_wav)
        ref_wavs.append(silence)

    ref_audio = np.concatenate(ref_wavs[:-1], axis=0)
    ref_output_path = os.path.join('Outputs', config['output_path']).replace('.wav', '_ref.wav')
    os.makedirs(os.path.dirname(ref_output_path), exist_ok=True)
    sf.write(ref_output_path, ref_audio, 24000, format='WAV', subtype='PCM_16')



    wavs_syn = []
    wavs_comb = []
    silence = np.zeros(int(24000 * 0.5)) # 0.5 sec silence for interval
    output_path = os.path.join('Outputs', config['output_path'])
    output_path_comb = output_path.replace('.wav', '_comb.wav')


    loop_map = {'model': models, 'text': texts, 'reference': refs}
    loop_order = config['loop_order']
    for loop1 in loop_map[loop_order[0]]:
        for loop2 in loop_map[loop_order[1]]:
            s_prev = None
            for loop3 in loop_map[loop_order[2]]:
                model, model_params, sampler = loop1 if loop_order[0] == 'model' else loop2 if loop_order[1] == 'model' else loop3
                tokens = loop1 if loop_order[0] == 'text' else loop2 if loop_order[1] == 'text' else loop3
                ref_wav_path, ref_s, alpha, beta, diffusion_steps, embedding_scale = loop1 if loop_order[0] == 'reference' else loop2 if loop_order[1] == 'reference' else loop3

                start = time.time()
                if loop_order[2] == 'reference':
                    ref_wav = librosa.load(ref_wav_path, sr=24000)[0]
                    wavs_comb.append(ref_wav)
                    wavs_comb.append(silence)
                if loop_order[2] == 'text':
                    # Long Form inference
                    wav, s_prev = inference(model, model_params, sampler, tokens, s_prev, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
                else:
                    wav, s_prev = inference(model, model_params, sampler, tokens, None, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
                wavs_syn.append(wav)
                wavs_syn.append(silence)
                wavs_comb.append(wav)
                wavs_comb.append(silence)

                rtf = (time.time() - start) / (len(wav) / 24000)
                print(f"RTF = {rtf:5f}")

    audio_syn = np.concatenate(wavs_syn[:-1], axis=0)
    audio_comb = np.concatenate(wavs_comb[:-1], axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_syn, 24000, format='WAV', subtype='PCM_16')
    sf.write(output_path_comb, audio_comb, 24000, format='WAV', subtype='PCM_16')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='StyleTTS2 Inference')
parser.add_argument('-c', '--config_path', type=str, default='Inference/reference_test.yml', help='path to the config file for inference')
args = parser.parse_args()

main(args)