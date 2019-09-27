import traceback
from collections import defaultdict
from functools import partial
import torch
from torch import nn

from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.train_stats = {}
        self.eval_stats = {}
        self.parameter_reference = None
        self.activation_means = defaultdict(list)
        self.activation_stds = defaultdict(list)
        self.activation_names = []
        self.hook_handles = []

    def save_parameters_for_reference(self, model):
        self.parameter_reference = {name: param.clone()
                                    for name, param in model.named_parameters()}

    def install_activation_hooks(self, model):
        if len(self.hook_handles) != 0:
            raise RuntimeError("Can't install new hooks before "
                               "releasing old ones")

        def _activation_hook(module, in_tensor, out_tensor, name):
            with torch.no_grad():
                if name == "encoder.lstm":
                    out_tensor = out_tensor[0][0].cpu()
                    self.activation_means[name].append(out_tensor.mean(dim=1))
                    self.activation_stds[name].append(out_tensor.std(dim=1))
                elif name == "gst.encoder.recurrence":
                    out_tensor = out_tensor[0][0].cpu()
                    self.activation_means[name].append(out_tensor.mean(dim=1))
                    self.activation_stds[name].append(out_tensor.std(dim=1))
                elif name == "decoder.attention_rnn" or name == "decoder.decoder_rnn":
                    if isinstance(module, nn.LSTMCell):
                        out_tensor = (out_tensor[0].cpu(),
                                      out_tensor[1].cpu())

                        self.activation_means[f"{name}.hidden"].append(out_tensor[0].mean(dim=1))
                        self.activation_means[f"{name}.cell"].append(out_tensor[1].mean(dim=1))
                        self.activation_stds[f"{name}.hidden"].append(out_tensor[0].std(dim=1))
                        self.activation_stds[f"{name}.cell"].append(out_tensor[1].std(dim=1))
                    else:
                        self.activation_means[f"{name}.hidden"].append(
                            out_tensor.mean(dim=1))
                        self.activation_stds[f"{name}.hidden"].append(
                            out_tensor.std(dim=1))
                else:
                    out_tensor = out_tensor.cpu()
                    if "convolutions" in name:
                        out_tensor = out_tensor.transpose(1, 2)
                    out_tensor = out_tensor.reshape((-1, out_tensor.size(-1)))
                    self.activation_means[name].append(out_tensor.mean(dim=1))
                    self.activation_stds[name].append(out_tensor.std(dim=1))

        def hook_all(layer, names):
            if len(layer._modules) == 0 \
                    or isinstance(layer, torch.nn.Sequential):
                name = ".".join(names)
                # print(name)
                if name.endswith("_rnn"):
                    self.activation_names.append(f"{name}.hidden")
                    if isinstance(layer, nn.LSTMCell):
                        self.activation_names.append(f"{name}.cell")
                else:
                    self.activation_names.append(name)
                self.hook_handles.append(
                    layer.register_forward_hook(
                        partial(_activation_hook, name=name)
                    )
                )
            else:
                for name, next_layer in layer._modules.items():
                    hook_all(next_layer, names + [name])

        hook_all(model, [])

    def remove_activation_hooks(self):
        for hook in self.hook_handles:
            hook.remove()

        self.activation_means = defaultdict(list)
        self.activation_stds = defaultdict(list)
        self.activation_names = []
        self.hook_handles = []

    def _concat_activation_stats(self):
        activation_stds = {name: torch.cat(t, dim=0)
                              for name, t in self.activation_stds.items()}
        activation_means = {name: torch.cat(t, dim=0)
                            for name, t in self.activation_means.items()}
        return activation_means, activation_stds

    def tb_model_weights(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_scalar(
                f"{name}/param-max",
                param.max(), step)
            self.writer.add_scalar(
                f"{name}/param-min",
                param.min(), step)
            self.writer.add_scalar(
                f"{name}/param-mean",
                param.mean(), step)
            self.writer.add_scalar(
                f"{name}/param-std",
                param.std(), step)
            self.writer.add_histogram(
                f"{name}/param", param, step)
            self.writer.add_histogram(
                f"{name}/grad", param.grad, step)
            self.writer.add_histogram(
                f"{name}/param-movement",
                param - self.parameter_reference[name],
                step)

    def log_activation_stats(self, step, suffix):
        if len(self.activation_names) > 0:
            activation_means, activation_stddevs = \
                self._concat_activation_stats()
            for name in self.activation_names:
                self.writer.add_histogram(
                    f"{name}/activation-means-{suffix}",
                    activation_means[name], step)
                if not torch.isnan(activation_stddevs[name]).any():
                    self.writer.add_histogram(
                        f"{name}/activation-stddevs-{suffix}",
                        activation_stddevs[name], step)

    def dict_to_tb_scalar(self, scope_name, stats, step):
        for key, value in stats.items():
            self.writer.add_scalar('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_figure(self, scope_name, figures, step):
        for key, value in figures.items():
            self.writer.add_figure('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            try:
                self.writer.add_audio('{}/{}'.format(scope_name, key), value, step, sample_rate=sample_rate)
            except:
                traceback.print_exc()

    def tb_train_iter_stats(self, step, stats):
        self.dict_to_tb_scalar("TrainIterStats", stats, step)

    def tb_train_epoch_stats(self, step, stats):
        self.dict_to_tb_scalar("TrainEpochStats", stats, step)

    def tb_train_figures(self, step, figures):
        self.dict_to_tb_figure("TrainFigures", figures, step)

    def tb_train_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("TrainAudios", audios, step, sample_rate)

    def tb_eval_stats(self, step, stats):
        self.dict_to_tb_scalar("EvalStats", stats, step)

    def tb_eval_figures(self, step, figures):
        self.dict_to_tb_figure("EvalFigures", figures, step)

    def tb_eval_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("EvalAudios", audios, step, sample_rate)

    def tb_test_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("TestAudios", audios, step, sample_rate)

    def tb_test_figures(self, step, figures):
        self.dict_to_tb_figure("TestFigures", figures, step)
