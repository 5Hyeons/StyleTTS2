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

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
from nltk.tokenize import word_tokenize
# IPA Phonemizer: https://github.com/bootphon/phonemizer
from sentence_transformers import SentenceTransformer

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
textclenaer = TextCleaner()
text_embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cpu')

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
            # try:
            #     model[key].load_state_dict(params[key], strict=True)
            # except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            print(f'{key} key lenghth: {len(model[key].state_dict().keys())}, state_dict length: {len(state_dict.keys())}')
            for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                new_state_dict[k_m] = v_c
            model[key].load_state_dict(new_state_dict, strict=True)
            model[key].eval()
            model[key].to(device)
            print('%s loaded' % key)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    return model, model_params, sampler

def inference(model, model_params, sampler, text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, history=None):
    text = text.strip()
    text_emb = text_embedder.encode([text])
    history['text'].append(torch.tensor(text_emb).to(device))
    # ps = text.strip()
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

        h_bert = model.bert(tokens, attention_mask=(~text_mask).int())
        # history['text'].append(h_bert.detach())

        if len(history['text']) > 2:
            data = HeteroData()
            text_tensor = torch.cat(history['text'], dim=0)
            acoustic_tensor = torch.cat(history['acoustic'], dim=0)
            prosody_tensor = torch.cat(history['prosody'], dim=0)

            data["text"].x = text_tensor
            data["acoustic"].x = acoustic_tensor
            data["prosody"].x = prosody_tensor

            edge = []
            for _i in range(data["prosody"].x.shape[0]):
                for _j in range(data["acoustic"].x.shape[0]):
                    edge.append([_j, _i])
            data["acoustic", "to", "prosody"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
            data["acoustic", "to", "acoustic"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
            data["prosody", "to", "prosody"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)

            edge = []
            # the length of the text is one more than the length of the acoustic/prosodic features
            for _i in range(data["text"].x.shape[0]):
                for _j in range(data["acoustic"].x.shape[0]):
                    edge.append([_j, _i])
            data["acoustic", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
            data["prosody", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)

            edge = []
            for _i in range(data["text"].x.shape[0]):
                for _j in range(data["text"].x.shape[0]):
                    edge.append([_j, _i])
            data["text", "to", "text"].edge_index = torch.tensor(edge).contiguous().transpose(-2, -1)
            data = T.ToUndirected()(data)

            data, model.hgt = data.to(device), model.hgt.to(device)
            out_text = model.hgt(data.x_dict, data.edge_index_dict)
            current_text_tensor = history['text'][-1]
            q = current_text_tensor.unsqueeze(0)
            k = v = out_text[:-1].unsqueeze(0)
            s_conv = model.style_predictor(q, k, v)[0] # [1, 1, 256]

        else:
            s_conv = torch.zeros((1, 1, 256)).to(device)
            print('at first')
        # d_en = model.bert_encoder(h_bert).transpose(-1, -2) 
        h_bert = torch.cat([h_bert, s_conv.expand(-1, h_bert.size(1), -1)], dim=-1)
        d_en = model.bert_encoder(h_bert).transpose(-1, -2)
        
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=h_bert,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        # ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        # s = beta * s + (1 - beta) * ref_s[:, 128:]

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
    return out.squeeze().cpu().numpy()[..., :-50], intervals, s_pred # weird pulse at the end of the model, need to be fixed later


def main(args):
    model_name = os.path.dirname(args.model_path).split('/')[-1]
    config_path = os.path.dirname(args.model_path) + '/config_' + model_name + '.yml'
    config = yaml.safe_load(open(config_path))

    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    model, model_params, sampler = get_model(config, args.model_path, plbert)

    texts = []
    sids = []
    ori_wav_paths = []
    if args.texts:
        with open(args.texts, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "" or "#" in line:
                    continue
                if line =='```':
                    break
                texts.append(line)
                ori_wav_paths.append(line.split('|')[0])
        output_path = os.path.join('Outputs', args.model_path.split('/')[1], args.texts.split('/')[-1].replace('.txt', '.wav'))
    elif args.dataset and args.num:
        dataset = args.dataset
        num = args.num
        dialogue_dir = os.path.join(dataset, num)
        for file in sorted(os.listdir(dialogue_dir), key=lambda x: int(x.split('_')[0])):
            if file.endswith('.txt'):
                with open(os.path.join(dialogue_dir, file), 'r') as f:
                    line = f.readline().strip()
                    texts.append(line)
            elif file.endswith('.wav'):
                ori_wav_paths.append(os.path.join(dialogue_dir, file))
                sid = file.split('_')[1]
                sids.append(sid)
        output_path = os.path.join('Outputs', args.model_path.split('/')[1], f'd_{num}.wav')
    else:
        texts.append(args.text)
        output_path = os.path.join('Outputs', args.model_path.split('/')[1], 'output.wav')


    wavs = []
    silence = np.zeros(int(24000 * 0.5)) # 0.5 sec silence for interval
    wavs.append(silence)

    ori_wavs = []
    for ori_wav_path in ori_wav_paths:
        audio, sr = librosa.load(ori_wav_path, sr=24000)
        audio, index = librosa.effects.trim(audio, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        audio = np.concatenate([np.zeros([5000]), audio, np.zeros([5000])], axis=0)
        ori_wavs.append(audio)
        ori_wavs.append(silence)
    ori_audio = np.concatenate(ori_wavs[:-1], axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path.replace('.wav', '_ori.wav'), ori_audio, 24000, format='WAV', subtype='PCM_16')

    history = {
        "text": [],
        "acoustic": [],
        "prosody": [],
    }

    dialouge_len = len(texts)
    for i, text in enumerate(texts):
        if args.half and i < dialouge_len//2:
            text_emb = text_embedder.encode([text.strip()])
            history['text'].append(torch.tensor(text_emb).to(device))
            wav_path = ori_wav_paths[i]
            style = compute_style(wav_path, model)
            history['acoustic'].append(style[:, :128])
            history['prosody'].append(style[:, 128:])
            print(f'continue: {i+1}/{dialouge_len}')
            continue

        start = time.time()
        # _, text, ref_wav_path = text.split('|')
        # text, ref_wav_path = text.split('|')
        if sids[i] == '0':
            ref_wav_path = 'wavs/dailytalk/72/1_0_d72.wav'
        elif sids[i] == '1':
            ref_wav_path = 'wavs/dailytalk/79/4_1_d79.wav'
        ref_s = compute_style(ref_wav_path, model)
        wav, intervals, s_pred = inference(model, model_params, sampler, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1, history=history)
        s_a = s_pred[:, :128]
        s_p = s_pred[:, 128:]
        history['acoustic'].append(s_a)
        history['prosody'].append(s_p)
        wavs.append(wav)
        wavs.append(silence)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")

    audio = np.concatenate(wavs[:-1], axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, 24000, format='WAV', subtype='PCM_16')
    print('Saved to', output_path)
    create_textgrid_from_tokens(intervals, output_path.replace('.wav', '.TextGrid'))


parser = argparse.ArgumentParser(description='StyleTTS2 Inference')
# parser.add_argument('-c', '--config_path', type=str, default='/home/jovyan/code/StyleTTS2/Models/dailytalk_conv_back/config_dailytalk_conv.yml', help='path to the config file')
parser.add_argument('-m', '--model_path', type=str, default='Models/dailytalk_conversational/epoch_2nd_00090.pth', help='path to the model')
parser.add_argument('-d', '--dataset', type=str, default='wavs/dailytalk', help='dataset name')
parser.add_argument('-n', '--num', type=str, default='0', help='dialogue number')
parser.add_argument('-t', '--text', type=str, default='Fluent was founded in 2021, and is a company that develops technologies that express movements of Generative AI. Currently, the company is focusing on developing TalkMotion AI, an interactive AI virtual human solution.', help='text to synthesize')
parser.add_argument('--texts', type=str, default='', help='path to the text file to synthesize')
parser.add_argument('--half', action='store_true', help='inference from the center of the dialogue')
args = parser.parse_args()

main(args)