import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets.TTSDataset import MyDataset
from distribute import (DistributedSampler, apply_gradient_allreduce,
                        init_distributed, reduce_tensor)
from layers.losses import L1LossMasked, MSELossMasked
from utils.audio import AudioProcessor
from utils.generic_utils import (NoamLR, check_update, count_parameters,
                                 create_experiment_folder, get_git_branch,
                                 load_config, remove_experiment_folder,
                                 save_best_model, save_checkpoint, weight_decay,
                                 set_init_dict, copy_config_file, setup_model,
                                 split_dataset)
from utils.logger import Logger
from utils.speakers import load_speaker_mapping, save_speaker_mapping, \
    get_speakers
from utils.synthesis import synthesis
from utils.text.symbols import phonemes, symbols
from utils.visual import plot_alignment, plot_spectrogram
from datasets.preprocess import get_preprocessor_by_name

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)

# keeprates=[1.0, 0.8, 0.65, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05]
keeprates = [1.0]


def setup_loader(ap, is_val=False, verbose=False):
    global meta_data_train
    global meta_data_eval
    if "meta_data_train" not in globals():
        if c.meta_file_train is not None:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_train)
        else:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path)
    if "meta_data_eval" not in globals() and c.run_eval:
        if c.meta_file_val is not None:
            meta_data_eval = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_val)
        else:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            c.r,
            c.text_cleaner,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            batch_group_size=c.batch_group_size * c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=True)
    return loader, dataset


def train(model, criterion, criterion_alignment, optimizer, optimizer_st, scheduler,
          ap, epoch, data_loader):
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.train()
    epoch_time = 0
    postnet_losses = []
    decoder_losses = []
    alignment_losses = []
    # stop_losses = []
    step_times = []
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    teacher_keep_rate = keeprates[min(epoch // 8, len(keeprates) - 1)]
    print(f"\n > keep rate in teacher forcing: ${teacher_keep_rate}")
    batch_n_iter = int(len(data_loader.dataset) / (c.batch_size * num_gpus))
    if c.tb_model_param_stats and args.rank == 0:
        tb_logger.save_parameters_for_reference(model)
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        speaker_names = data[2]
        linear_input = data[3] if c.model == "Tacotron" else None
        mel_input = data[4]
        mel_lengths = data[5]
        # stop_targets = data[6]
        alignment_targets = data[7]
        alignment_mask = data[8]
        avg_text_length = torch.mean(text_lengths.float())
        avg_spec_length = torch.mean(mel_lengths.float())

        if c.use_speaker_embedding:
            speaker_ids = [speaker_mapping[speaker_name]
                           for speaker_name in speaker_names]
            speaker_ids = torch.LongTensor(speaker_ids)
        else:
            speaker_ids = None

        # set stop targets view, we predict a single stop token per r frames prediction
        # stop_targets = stop_targets.view(text_input.shape[0],
        #                                  stop_targets.size(1) // c.r, -1)
        # stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

        current_step = num_iter + args.restore_step + \
            epoch * len(data_loader) + 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st:
            optimizer_st.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(non_blocking=True) if c.model == "Tacotron" else None
            # stop_targets = stop_targets.cuda(non_blocking=True)
            alignment_targets = alignment_targets.cuda(non_blocking=True)
            alignment_mask = alignment_mask.cuda(non_blocking=True)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.cuda(non_blocking=True)


        # forward pass model
        decoder_output, postnet_output, alignments = model(
            text_input, text_lengths, mel_input, speaker_ids=speaker_ids,
            teacher_keep_rate=teacher_keep_rate)

        alignments_masked = alignments * alignment_mask.float()
        alignments_sum_pred = torch.clamp(torch.sum(alignments_masked, dim=1), 0.15, 0.5)
        alignment_targets = torch.clamp(alignment_targets, 0.15, 0.5)

        # loss computation
        # stop_loss = criterion_st(stop_tokens, stop_targets) if c.stopnet else torch.zeros(1)
        alignment_loss = criterion_alignment(alignments_sum_pred,
                                             alignment_targets)

        if c.loss_masking:
            decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
            if c.model == "Tacotron":
                postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
            else:
                postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
        else:
            decoder_loss = criterion(decoder_output, mel_input)
            if c.model == "Tacotron":
                postnet_loss = criterion(postnet_output, linear_input)
            else:
                postnet_loss = criterion(postnet_output, mel_input)
        loss = decoder_loss + postnet_loss + \
               c.alignment_loss_reg * alignment_loss
        # if not c.separate_stopnet and c.stopnet:
        #     loss += c.stop_loss_adjustment * stop_loss

        loss.backward()
        optimizer, current_lr = weight_decay(optimizer, c.wd)
        grad_norm, _ = check_update(model, c.grad_clip)
        optimizer.step()

        # backpass and check the grad norm for stop loss
        # if c.separate_stopnet:
        #     stop_loss.backward()
        #     optimizer_st, _ = weight_decay(optimizer_st, c.wd)
        #     grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0)
        #     optimizer_st.step()
        # else:
        #     grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        # aggregate losses from processes
        if num_gpus > 1:
            postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
            decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
            loss = reduce_tensor(loss.data, num_gpus)
            # stop_loss = reduce_tensor(stop_loss.data, num_gpus) if c.stopnet else stop_loss

        postnet_losses.append(float(postnet_loss.item()))
        decoder_losses.append(float(decoder_loss.item()))
        alignment_losses.append(float(alignment_loss.item()))
        # stop_losses.append(stop_loss if isinstance(stop_loss, float) else float(
        #     stop_loss.item()))
        step_times.append(step_time)

        if current_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{} "
                "PostnetLoss:{:.5f}  "
                "DecoderLoss:{:.5f}  "
                "AlignmentLoss:{:.5f}  "
                "GradNorm:{:.5f}  "
                "AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  "
                "StepTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, current_step,
                    np.mean(postnet_losses[-c.print_step:]),
                    np.mean(decoder_losses[-c.print_step:]),
                    np.mean(alignment_losses[-c.print_step:]),
                    grad_norm, avg_text_length,
                    avg_spec_length, np.mean(step_times[-c.print_step:]),
                    current_lr),
                flush=True)

        if args.rank == 0:
            if current_step % c.print_step == 0:
            # Plot Training Iter Stats
                iter_stats = {"ma_loss_postnet": np.mean(postnet_losses[-c.print_step:]),
                            "ma_loss_decoder": np.mean(decoder_losses[-c.print_step:]),
                            "ma_loss_alignment": np.mean(alignment_losses[-c.print_step:]),
                            "lr": current_lr,
                            "grad_norm": grad_norm,
                            "ma_step_time": np.mean(step_times[-c.print_step:])}
                tb_logger.tb_train_iter_stats(current_step, iter_stats)

            if current_step % c.save_step == 0 and c.checkpoint:
                # save model
                save_checkpoint(model, optimizer, optimizer_st,
                                postnet_loss.item(), OUT_PATH, current_step,
                                epoch)

            if current_step % c.train_diag_step == 0:
                # Diagnostic visualizations
                const_spec = postnet_output[-1].data.cpu().numpy()
                gt_spec = linear_input[-1].data.cpu().numpy() if c.model == "Tacotron" else mel_input[-1].data.cpu().numpy()
                align_img = alignments[-1].data.cpu().numpy()
                text_len = text_lengths[-1].data.cpu().numpy()
                spec_len = mel_lengths[-1].data.cpu().numpy()
                gt_spec = gt_spec[:spec_len]
                const_spec = const_spec[:spec_len]

                figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img,
                                                text_padding_start=text_len,
                                                spec_padding_start=spec_len // c.r)
                }
                tb_logger.tb_train_figures(current_step, figures)

                # Sample audio
                if c.model == "Tacotron":
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_train_audios(current_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])

    total_losses = [decoder_losses[i] + postnet_losses[i] + alignment_losses[i]
                    for i in range(len(postnet_losses))]

    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgPostnetLoss:{:.5f}  AvgDecoderLoss:{:.5f}  "
        "AvgAlignmentLoss:{:.5f}  "
        "EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}".format(current_step, np.mean(total_losses),
                                    np.mean(postnet_losses),
                                    np.mean(decoder_losses),
                                    np.mean(alignment_losses),
                                    epoch_time,
                                    np.mean(step_times)),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {"loss_postnet": np.mean(postnet_losses),
                    "loss_decoder": np.mean(decoder_losses),
                    "epoch_time": epoch_time}
        tb_logger.tb_train_epoch_stats(current_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, current_step)

    return np.mean(postnet_losses), current_step


def evaluate(model, criterion, criterion_alignment, ap, current_step, epoch,
             data_loader, mode="easy"):
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.eval()
    epoch_time = 0
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_alignment_loss = 0
    teacher_keep_rate = 1.0 if mode == "easy" else 0.5
    print(f"\n > Validation: {mode}")
    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                start_time = time.time()

                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                speaker_names = data[2]
                linear_input = data[3] if c.model == "Tacotron" else None
                mel_input = data[4]
                mel_lengths = data[5]
                # stop_targets = data[6]
                alignment_targets = data[7]
                alignment_mask = data[8]

                if c.use_speaker_embedding:
                    speaker_ids = [speaker_mapping[speaker_name]
                                   for speaker_name in speaker_names]
                    speaker_ids = torch.LongTensor(speaker_ids)
                else:
                    speaker_ids = None

                # set stop targets view, we predict a single stop token per r frames prediction
                # stop_targets = stop_targets.view(text_input.shape[0],
                #                                  stop_targets.size(1) // c.r,
                #                                  -1)
                # stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda()
                    mel_input = mel_input.cuda()
                    mel_lengths = mel_lengths.cuda()
                    linear_input = linear_input.cuda() if c.model == "Tacotron" else None
                    # stop_targets = stop_targets.cuda()
                    alignment_targets = alignment_targets.cuda()
                    alignment_mask = alignment_mask.cuda()
                    if speaker_ids is not None:
                        speaker_ids = speaker_ids.cuda()

                # forward pass
                decoder_output, postnet_output, alignments =\
                    model.forward(text_input, text_lengths, mel_input,
                                  speaker_ids=speaker_ids,
                                  teacher_keep_rate=teacher_keep_rate)

                alignments_masked = alignments * alignment_mask.float()
                alignments_sum_pred = torch.clamp(
                    torch.sum(alignments_masked, dim=1), 0.15, 0.5)
                alignment_targets = torch.clamp(alignment_targets, 0.15, 0.5)

                # loss computation
                # stop_loss = criterion_st(stop_tokens, stop_targets) if
                # c.stopnet else torch.zeros(1)
                alignment_loss = criterion_alignment(alignments_sum_pred,
                                                     alignment_targets)
                if c.loss_masking:
                    decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
                    if c.model == "Tacotron":
                        postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
                    else:
                        postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
                else:
                    decoder_loss = criterion(decoder_output, mel_input)
                    if c.model == "Tacotron":
                        postnet_loss = criterion(postnet_output, linear_input)
                    else:
                        postnet_loss = criterion(postnet_output, mel_input)
                loss = decoder_loss + postnet_loss + alignment_loss

                step_time = time.time() - start_time
                epoch_time += step_time

                if num_iter % c.print_step == 0:
                    print(
                        "   | > TotalLoss: {:.5f}   PostnetLoss: {:.5f}   "
                        "DecoderLoss:{:.5f}  "
                        "AlignmentLoss:{:.5f}".format(loss.item(),
                                                      postnet_loss.item(),
                                                      decoder_loss.item(),
                                                      alignment_loss.item()),
                        flush=True)

                # aggregate losses from processes
                if num_gpus > 1:
                    postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
                    decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
                    # if c.stopnet:
                    #     stop_loss = reduce_tensor(stop_loss.data, num_gpus)

                avg_postnet_loss += float(postnet_loss.item())
                avg_decoder_loss += float(decoder_loss.item())
                avg_alignment_loss += alignment_loss.item()

            if args.rank == 0:
                # Diagnostic visualizations
                idx = np.random.randint(mel_input.shape[0])
                const_spec = postnet_output[idx].data.cpu().numpy()
                gt_spec = linear_input[idx].data.cpu().numpy() if c.model == "Tacotron" else mel_input[idx].data.cpu().numpy()
                align_img = alignments[idx].data.cpu().numpy()
                text_len = text_lengths[idx].data.cpu().numpy()
                spec_len = mel_lengths[idx].data.cpu().numpy()
                gt_spec = gt_spec[:spec_len]
                const_spec = const_spec[:spec_len]

                eval_figures = {
                    f"prediction ({mode})": plot_spectrogram(const_spec, ap),
                    f"ground_truth ({mode})": plot_spectrogram(gt_spec, ap),
                    f"alignment ({mode})": plot_alignment(align_img,
                                                text_padding_start=text_len,
                                                spec_padding_start=spec_len // c.r)
                }
                tb_logger.tb_eval_figures(current_step, eval_figures)

                # Sample audio
                if c.model == "Tacotron":
                    eval_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    eval_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_eval_audios(current_step, {"ValAudio": eval_audio}, c.audio["sample_rate"])

                # compute average losses
                avg_postnet_loss /= (num_iter + 1)
                avg_decoder_loss /= (num_iter + 1)
                avg_alignment_loss /= (num_iter + 1)

                # Plot Validation Stats
                epoch_stats = {f"loss_postnet ({mode})": avg_postnet_loss,
                               f"loss_decoder ({mode})": avg_decoder_loss,
                               f"alignment_loss ({mode})": avg_alignment_loss}
                tb_logger.tb_eval_stats(current_step, epoch_stats)

    return avg_postnet_loss


def test(model, criterion, criterion_alignment, ap, current_step, epoch):
    if c.test_sentences_file is None:
        test_sentences = [
            # "Am Ende gibt es nur einen Verlierer: die SPD. Die Sozialdemokraten haben mit ihrem 'Nein' zu Ursula von der Leyen weder sich noch dem Spitzenkandidatenprinzip geholfen.",
            "In Kitas, Schulen und Asylunterkünften soll künftig ein Impfschutz verpflichtend sein, auch für Angestellte.",
            "Es tut mir leid Thomas, aber ich kann das leider nicht tun.",
            "Auf den sieben Robbenklippen sitzen sieben Robbensippen, die sich in die Rippen stippen, bis sie von den Klippen kippen.",
            # "Ich esse meine Suppe nicht! Nein, meine Suppe ess' ich nicht!",
            # "Hallo Frau Peters, ich bin Herr Müller.",
            # "Da zogen sie sich ihre besten Kleider an und gingen in die große weiße Kirche.",
            # "Soll mich doch wundern, wo der Bengel wieder steckt! Tom!"
        ]
    else:
        with open(c.test_sentences_file, "r") as f:
            test_sentences = [s.strip() for s in f.readlines()]
    if args.rank == 0 and epoch > c.test_delay_epochs:
        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        style_wav = c.get("style_wav_for_test")

        if c.use_speaker_embedding:
            speaker_mapping = load_speaker_mapping(OUT_PATH)
            for speaker_name, speaker_id in speaker_mapping.items():
                for idx, test_sentence in enumerate(test_sentences):
                    try:
                        wav, alignment, decoder_output, postnet_output \
                            = synthesis(
                            model, test_sentence, c, use_cuda, ap,
                            speaker_id=speaker_id,
                            style_wav=style_wav)
                        file_path = os.path.join(AUDIO_PATH, str(current_step))
                        os.makedirs(file_path, exist_ok=True)
                        file_path = os.path.join(file_path,
                                                 "TestSentence_{}-{}.wav".format(
                                                     speaker_name, idx))
                        ap.save_wav(wav, file_path)
                        test_audios['{}-{}-audio'.format(speaker_name, idx)] = wav
                        test_figures[
                            '{}-{}-prediction'.format(speaker_name, idx)] = plot_spectrogram(
                            postnet_output, ap)
                        test_figures[
                            '{}-{}-alignment'.format(speaker_name, idx)] = plot_alignment(
                            alignment)
                    except:
                        print(" !! Error creating Test Sentence -", idx)
                        traceback.print_exc()
        else:
            speaker_id = None
            for idx, test_sentence in enumerate(test_sentences):
                try:
                    wav, alignment, decoder_output, postnet_output = synthesis(
                        model, test_sentence, c, use_cuda, ap,
                        speaker_id=speaker_id,
                        style_wav=style_wav)
                    file_path = os.path.join(AUDIO_PATH, str(current_step))
                    os.makedirs(file_path, exist_ok=True)
                    file_path = os.path.join(file_path,
                                             "TestSentence_{}.wav".format(idx))
                    ap.save_wav(wav, file_path)
                    test_audios['{}-audio'.format(idx)] = wav
                    test_figures['{}-prediction'.format(idx)] = plot_spectrogram(postnet_output, ap)
                    test_figures['{}-alignment'.format(idx)] = plot_alignment(alignment)
                except:
                    print(" !! Error creating Test Sentence -", idx)
                    traceback.print_exc()
        tb_logger.tb_test_audios(current_step, test_audios, c.audio['sample_rate'])
        tb_logger.tb_test_figures(current_step, test_figures)


#FIXME: move args definition/parsing inside of main?
def main(args): #pylint: disable=redefined-outer-name
    # Audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    if c.use_speaker_embedding:
        speakers = get_speakers(c.data_path, c.meta_file_train, c.dataset)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i
                               for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model = setup_model(num_chars, num_speakers, c)

    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=0)
    # if c.stopnet and c.separate_stopnet:
    #     optimizer_st = optim.Adam(
    #         model.decoder.stopnet.parameters(), lr=c.lr, weight_decay=0)
    # else:
    #     optimizer_st = None

    if c.loss_masking:
        criterion = L1LossMasked() if c.model == "Tacotron" else MSELossMasked()
    else:
        criterion = nn.L1Loss() if c.model == "Tacotron" else nn.MSELoss()
    criterion_alignment = nn.L1Loss()

    if c.get("combine_loss", False):
        l1 = nn.L1Loss()
        l2 = nn.MSELoss()
        criterion = lambda pred, target: l1(pred, target) + l2(pred, target)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model = model.cuda()
        criterion_alignment.cuda()
        if c.get("combine_loss", False):
            l1.cuda()
            l2.cuda()
        else:
            criterion.cuda()

        # if criterion_st:
        #     criterion_st.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.lr_decay:
        scheduler = NoamLR(
            optimizer,
            warmup_steps=c.warmup_steps,
            last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    train_loader, train_dataset = setup_loader(ap, is_val=False, verbose=True)
    eval_loader, eval_dataset = setup_loader(ap, is_val=True)

    for epoch in range(0, c.epochs):
        train_loss, current_step = train(model, criterion, criterion_alignment,
                                         optimizer, None, scheduler,
                                         ap, epoch, train_loader)
        val_loss = evaluate(model, criterion, criterion_alignment, ap, current_step, epoch, eval_loader)
        # val_loss_hard = evaluate(model, criterion, criterion_alignment, ap, current_step, epoch, "hard")
        # print(
        #     " | > Training Loss: {:.5f}   Validation Loss: (easy) {:.5f}   "
        #     "Validation Loss: (hard) {:.5f}".format(
        #         train_loss, val_loss, val_loss_hard),
        #     flush=True)
        print(
            " | > Training Loss: {:.5f}   "
            "Validation Loss: (easy) {:.5f}".format(train_loss, val_loss),
            flush=True)
        test(model, criterion, criterion_alignment, ap, current_step, epoch)
        target_loss = train_loss
        if c.run_eval:
            target_loss = val_loss
        best_loss = save_best_model(model, optimizer, target_loss, best_loss,
                                    OUT_PATH, current_step, epoch)
        train_dataset.verbose = False
        train_dataset.make_index()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='folder name for traning outputs.'
    )

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument(
        '--group_id',
        type=str,
        default="",
        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '' and args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path, os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    if args.rank == 0:
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) #pylint: disable=protected-access
    except Exception: #pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
