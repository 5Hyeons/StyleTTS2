# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

# from Utils.ASR.models import ASRCNN
# from Utils.JDC.model import JDCNet
# from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

# from Modules.slmadv import SLMAdversarialLoss
# from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

import pdb


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    
    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    # diff_epoch = loss_params.diff_epoch
    # joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    # BERT_path = config.get('PLBERT_dir', False)
    # plbert = load_plbert(BERT_path, use_checkpoint=False)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    # model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    model = build_model(model_params, text_aligner, pitch_extractor, None)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            epochs += start_epoch
        else:
            raise ValueError('You need to specify the path to the first stage model.') 
    
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    # wl = WavLMLoss(model_params.slm.model, 
    #                model.wd, 
    #                sr, 
    #                model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    # wl = MyDataParallel(wl)
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
                                    model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True),
                                    ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'wd', 'diffusion']
                                    ) # keep starting epoch for tensorboard log
                                    
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    iters = 0
    
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    # print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    # start_ds = False
    start_ds = True
    
    running_std = []
    
    # slmadv_params = Munch(config['slmadv_params'])
    # slmadv = SLMAdversarialLoss(model, wl, sampler, 
    #                             slmadv_params.min_len, 
    #                             slmadv_params.max_len,
    #                             batch_percentage=slmadv_params.batch_percentage,
    #                             skip_update=slmadv_params.iter, 
    #                             sig=slmadv_params.sig
    #                            )


    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.msd.train()
        model.mpd.train()
        model.mel_encoder.train()
        model.pitch_extractor.bilstm_classifier.train()


        # if epoch >= diff_epoch:
        #     start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    print('Error in alignment !!!')
                    continue

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)
                ph_lables = (texts.unsqueeze(1).float() @ s2s_attn_mono) # phoneme lables
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            ph = []
            # p_en = []
            # wav = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                # p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                ph.append(ph_lables[bib, :, random_start:random_start+mel_len])

                # y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                # wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
            # wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            # p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            ph = torch.stack(ph).detach()
            
            if gt.size(-1) < 80:
                continue

            # reconstruction
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            
                s_org = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
                s_trg = model.style_encoder(ref_mels.unsqueeze(1))
                
                y_gt = model.decoder(en, F0_real, N_real, s_org)
                y_fake_gt = model.decoder(en, F0_real, N_real, s_trg)
                
                # if epoch >= joint_epoch:
                    # ground truth from recording
                    # wav = y_rec_gt # use recording since decoder is tuned
                # else:
                    # ground truth from reconstruction
                    # wav = y_rec_gt_pred # use reconstruction since decoder is fixed

            en_mel = model.mel_encoder(gt, is_mel=True)
            ph_pred = model.linear_proj(en_mel)

            # phoneme information maximazation loss
            loss_pim = 0
            for k in range(0, ph_pred.shape[-1]):
                loss_pim += F.cross_entropy(ph_pred[:, :, k], ph.squeeze()[:, k].long())
            loss_pim /= ph_pred.shape[-1]

            y_rec = model.decoder(en_mel, F0_real, N_real, s_trg)
            mel_rec = to_mel(y_rec)
            # pdb.set_trace()
            
            # discriminator loss
            if start_ds:
                optimizer.zero_grad()
                y_fake_gt.requires_grad_()
                d_loss = dl(y_fake_gt, y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd')
                optimizer.step('mpd')
            else:
                d_loss = 0

            # generator loss
            optimizer.zero_grad()

            en_cyc = model.mel_encoder(y_rec, is_mel=False)
            F0_fake, _, _ = model.pitch_extractor(mel_rec.unsqueeze(1))
            N_fake = log_norm(mel_rec.unsqueeze(1)).squeeze(1)
            # pdb.set_trace()
            y_cyc = model.decoder(en_cyc, F0_fake, N_fake, s_org)

            loss_cyc = stft_loss(y_cyc.squeeze(), y_gt)
            loss_adv_cyc = dl(y_gt, y_cyc).mean()

            # adversarial feature matching loss
            if start_ds:
                loss_gen_all = gl(y_fake_gt, y_rec).mean()
            else:
                loss_gen_all = 0

            # mel reconstruction loss
            loss_mel = stft_loss(y_rec, y_fake_gt)

            # slm loss
            # loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_pim * loss_pim + \
                     loss_params.lambda_cyc * loss_cyc + \
                     loss_params.lambda_adv * loss_adv_cyc
                    #  loss_params.lambda_slm * loss_lm

            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            # optimizer.step('bert_encoder')
            # optimizer.step('bert')
            # optimizer.step('prosodic_text_encoder')
            # optimizer.step('predictor')
            # optimizer.step('predictor_encoder')
            optimizer.step('mel_encoder')
            optimizer.step('linear_proj')
            
            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Gen Loss: %.5f, PIM Loss: %.5f, Cycle Loss: %.5f, Cycle adv Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_gen_all, loss_pim, loss_cyc, loss_adv_cyc))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/pim_loss', loss_pim, iters)
                writer.add_scalar('train/cyc_loss', loss_cyc, iters)
                
                running_loss = 0
                print('Time elasped:', time.time()-start_time)
                
        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()
                
                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    # p_en = []
                    # wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        # y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        # wav.append(torch.from_numpy(y).to(device))

                    # wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    gt = torch.stack(gt).detach()

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    # F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()
                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                    # reconstruction
                    s = model.style_encoder(gt.unsqueeze(1))
                    mel_en = model.mel_encoder(gt, is_mel=True)
                    y_gt = model.decoder(en, F0_real, N_real, s)
                    y_rec = model.decoder(mel_en, F0_real, N_real, s)
                    mel_rec = to_mel(y_rec)
                    
                    loss_mel = stft_loss(y_gt.squeeze(), y_rec.detach())

                    loss_test += (loss_mel).mean()
                    iters_test += 1
                except Exception as e:
                    print(f"run into exception", e)
                    traceback.print_exc()
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        mel_image = get_image(mel_rec[0].cpu().numpy())
        writer.add_figure('eval/mel_rec', mel_image, epoch + 1)
        
        # generating reconstruction examples with GT duration
        with torch.no_grad():
            for bib in range(len(asr)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, :mel_length // 2].unsqueeze(0)
                ref_mel = ref_mels[bib, :, :mel_length].unsqueeze(0)

                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                F0_real = F0_real.unsqueeze(0)
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                s = model.style_encoder(gt.unsqueeze(1))
                s_trg = model.style_encoder(ref_mel.unsqueeze(1))
                mel_en = model.mel_encoder(gt, is_mel=True)

                y_gt = model.decoder(en, F0_real, N_real, s)
                y_fake_gt = model.decoder(en, F0_real, N_real, s_trg)
                y_rec = model.decoder(mel_en, F0_real, N_real, s_trg)

                writer.add_audio('gt/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                writer.add_audio('fake_gt/y' + str(bib), y_fake_gt.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                writer.add_audio('rec/y' + str(bib), y_gt.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                if bib >= 5:
                    break
                            
        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
            

if __name__=="__main__":
    main()
