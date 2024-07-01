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
from Utils.PLBERT.util import load_plbert

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
from nltk.tokenize import word_tokenize
# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
id_to_sym = {i: sym for i, sym in enumerate(symbols)}

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

textclenaer = TextCleaner()

def create_textgrid_from_tokens(tokens, output_path):
    # NLTK does not provide direct TextGrid support, so we manually create the content

    textgrid_content = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
    textgrid_content += 'xmin = 0\n'
    textgrid_content += f'xmax = {tokens[-1][-2]}\n'
    textgrid_content += 'tiers? <exists>\nsize = 1\nitem []:\n'
    textgrid_content += '    item [1]:\n        class = "IntervalTier"\n'
    textgrid_content += '        name = "words"\n'
    textgrid_content += f'        xmin = 0\n        xmax = {tokens[-1][-2]}\n'
    textgrid_content += f'        intervals: size = {len(tokens)}\n'

    for i, (s, e, word) in enumerate(tokens):
        textgrid_content += f'        intervals [{i+1}]:\n'
        textgrid_content += f'            xmin = {s}\n'
        textgrid_content += f'            xmax = {e}\n'
        textgrid_content += f'            text = "{word}"\n'

    with open(output_path, 'w', encoding='UTF-8') as f:
        f.write(textgrid_content)

    return output_path

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
    audio, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(audio, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    audio = np.concatenate([np.zeros([5000]), audio, np.zeros([5000])], axis=0)
    
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def get_model(config, ckpt_path, plbert):

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, None, None, plbert)

    params_whole = torch.load(ckpt_path, map_location='cpu')
    params = params_whole['net']

    ignore_modules = ['text_aligner', 'pitch_extractor', 'mpd', 'msd', 'wd']
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

def inference(model, model_params, sampler, text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    # text = text.strip()

    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    # tokens.append(0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)

        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
        
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        intervals = []
            
        for i, (ph, dur) in enumerate(zip(tokens[0], pred_dur)):
            if not intervals:
                start = 0
            else:
                start = intervals[-1][1]
            end = start + (dur * 300) / 24000 * 2
            end = round(end.item(), 4)
            
            token = id_to_sym[ph.item()]
            # if token == 'ᆫ' and dur > 15:
            #     pred_dur[i] = 5
            intervals.append((start, end, token))


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
    return out.squeeze().cpu().numpy()[..., :-50], intervals # weird pulse at the end of the model, need to be fixed later


def main(args):
    config = yaml.safe_load(open(args.config_path))

    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    model, model_params, sampler = get_model(config, args.model_path, plbert)

    texts = []
    if args.texts:
        with open(args.texts, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "" or "#" in line:
                    continue
                if line =='```':
                    break
                texts.append(line)
        output_path = os.path.join('Outputs', args.model_path.split('/')[1], args.texts.split('/')[-1].replace('.txt', '.wav'))
    else:
        texts.append(args.text)
        output_path = os.path.join('Outputs', args.model_path.split('/')[1], 'output.wav')

    ref_s = compute_style(args.ref_wav_path, model)

    wavs = []
    silence = np.zeros(int(24000 * 0.5)) # 0.5 sec silence for interval
    ref_wav, sr = librosa.load(args.ref_wav_path, sr=24000)
    wavs.append(ref_wav)
    wavs.append(silence)
    for i, text in enumerate(texts):
        start = time.time()
        wav, intervals = inference(model, model_params, sampler, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=16, embedding_scale=1)
        wavs.append(wav)
        wavs.append(silence)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")

    audio = np.concatenate(wavs[:-1], axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, 24000, format='WAV', subtype='PCM_16')
    create_textgrid_from_tokens(intervals, output_path.replace('.wav', '.TextGrid'))



device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='StyleTTS2 Inference')
parser.add_argument('-c', '--config_path', type=str, default='Models/dailytalk/config_ft_dailytalk.yml', help='path to the config file')
parser.add_argument('-m', '--model_path', type=str, default='Models/dailytalk/epoch_2nd_00009.pth', help='path to the model')
# parser.add_argument('-r', '--ref_wav_path', type=str, default='wavs/dailytalk/12/6_1_d12.wav', help='path to the reference wav file')
parser.add_argument('-r', '--ref_wav_path', type=str, default='wavs/dailytalk/22/0_0_d22.wav', help='path to the reference wav file')
parser.add_argument('-t', '--text', type=str, default='Fluent was founded in 2021, and is a company that develops technologies that express movements of Generative AI. Currently, the company is focusing on developing TalkMotion AI, an interactive AI virtual human solution.', help='text to synthesize')
parser.add_argument('--texts', type=str, default='', help='path to the text file to synthesize')
args = parser.parse_args()

main(args)