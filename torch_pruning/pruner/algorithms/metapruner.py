import torch
import torch.nn as nn
import typing

from .scheduler import linear_scheduler
from .. import function
from ... import ops, dependency
import numpy as np


class MetaPruner:
    """
    Meta Pruner for structural pruning.

    Args:
        model (nn.Module): A to-be-pruned model
        example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
        importance (Callable): importance estimator.
        global_pruning (bool): enable global pruning.
        ch_sparsity (float): global channel sparisty.
        ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
        iterative_steps (int): number of steps for iterative pruning.
        iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
        max_ch_sparsity (float): maximum channel sparsity.
        ignored_layers (List[nn.Module]): ignored modules.

        round_to (int): channel rounding.
        customized_pruners (dict): a dict containing module-pruner pairs.
        unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.
        root_module_types (list): types of prunable modules.
        output_transform (Callable): A function to transform network outputs.
    """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        global_pruning: bool = False,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None,
        # Advanced
        round_to: int = None,  # round channels to 8x, 16x, ...
        # for grouped channels.
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        # unwrapped nn.Parameters like ViT.pos_emb
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        root_module_types: typing.List = [
            ops.TORCH_CONV,
            ops.TORCH_LINEAR,
            ops.TORCH_LSTM,
        ],  # root module for each group
        output_transform: typing.Callable = None,
        optimal_transport: typing.Callable = None,
        backward_pruning=True,
        dimensionality_preserving=False,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning

        self.channel_groups = channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        self.backward_pruning = backward_pruning  # Added this
        self.dimensionality_preserving = dimensionality_preserving  # Added this

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
            backward_pruning=self.backward_pruning
        )

        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        self.optimal_transport = optimal_transport  # ADDED THIS

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The customized channel sparsity for different layers
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple(
                        [
                            ops.type2class(prunable_type)
                            for prunable_type in self.DG.REGISTERED_PRUNERS.keys()
                        ]
                    )
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[
                            submodule
                        ] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # detect group convs & group norms
        for m in self.model.modules():
            if (
                isinstance(m, ops.TORCH_CONV)
                and m.groups > 1
                and m.groups != m.out_channels
            ):
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups

        if self.global_pruning:
            initial_total_channels = 0
            for group in self.DG.get_all_groups(
                ignored_layers=self.ignored_layers,
                root_module_types=self.root_module_types,
            ):
                ch_groups = self.get_channel_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )
                initial_total_channels += (
                    self.DG.get_out_channels(
                        group[0][0].target.module) // ch_groups
                )
            self.initial_total_channels = initial_total_channels

        self.merge_bn()  # Added this

    def merge_bn(self):
        for _, m in self.model.named_modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                group = self.DG.get_pruning_group(
                    m, function.prune_batchnorm_in_channels, idxs=[0])
                for dep, _ in group:
                    if isinstance(dep.source.module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)) and \
                            isinstance(dep.target.module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                        conv = dep.target.module.weight
                        conv_bias = dep.target.module.bias if not dep.target.module.bias is None else 0.0
                        conv_mean = dep.source.module.running_mean
                        conv_var = torch.sqrt(dep.source.module.running_var)
                        conv_gamma = dep.source.module.weight
                        conv_beta = dep.source.module.bias

                        new_conv = (
                            conv / conv_var[:, None, None, None]) * conv_gamma[:, None, None, None]
                        new_conv_bias = (
                            (conv_bias - conv_mean)/conv_var)*conv_gamma + conv_beta

                        shape = conv_mean.shape[0]
                        dep.target.module.weight = torch.nn.Parameter(new_conv)
                        dep.target.module.bias = torch.nn.Parameter(
                            new_conv_bias)
                        dep.source.module.running_mean = torch.full(
                            (shape,), 0.0)
                        dep.source.module.running_var = torch.full(
                            (shape,), 1.0)
                        dep.source.module.weight = torch.nn.Parameter(
                            torch.full((shape,), 1.0))
                        dep.source.module.bias = torch.nn.Parameter(
                            torch.full((shape,), 0.0))

    def pruning_history(self):
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history):
        self.DG.load_pruning_history(pruning_history)

    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
            self.current_step
        ]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """Model regularizor"""
        pass

    def step(self, interactive=False, group_idxs=None):
        self.current_step += 1
        if self.global_pruning:
            if interactive:
                return self.prune_global()
            else:
                for group in self.prune_global():
                    group.prune()
        else:
            if interactive:
                return self.prune_local()
            else:
                for group in self.prune_local(group_idxs=group_idxs):
                    group.prune(
                        dimensionality_preserving=self.dimensionality_preserving)

    def estimate_importance(self, group, ch_groups=1):
        return self.importance(group, ch_groups=ch_groups)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None:
                    continue
                if (
                    layer_out_ch
                    < self.layer_init_out_ch[module] * (1 - self.max_ch_sparsity)
                    or layer_out_ch == 1
                ):
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None:
                    continue
                if (
                    layer_in_ch
                    < self.layer_init_in_ch[module] * (1 - self.max_ch_sparsity)
                    or layer_in_ch == 1
                ):
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping

    def build_feature_map(self, loader, group_idx, num_batches=5):
        feature_maps = {}
        for group in self.DG.get_all_groups_in_order(
            ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, group_idxs=[
                group_idx]
        ):
            feature_maps = {}
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                # Conv out_channels
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    def hook(self, input, output):
                        if not self in feature_maps:
                            feature_maps[self] = output[None,
                                                        :].clone().detach()
                        else:
                            feature_maps[self] = torch.cat(
                                (feature_maps[self], output[None, :].clone().detach()), dim=0)

                    handler = layer.register_forward_hook(hook)
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(loader):
                            # use 5 batches to get feature maps.
                            if batch_idx >= num_batches:
                                break
                            self.model(inputs)
                    handler.remove()

        return feature_maps

    def prune_local(self, group_idxs=None):
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups_in_order(
            ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, group_idxs=group_idxs
        ):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None:
                    continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] * (1 - target_sparsity)
                )

                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)

                if n_pruned <= 0:
                    continue
                if ch_groups > 1:
                    imp = imp[: len(imp) // ch_groups]
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[: (n_pruned // ch_groups)]
                if ch_groups > 1:
                    group_size = current_channels // ch_groups
                    pruning_idxs = torch.cat(
                        [pruning_idxs + group_size *
                            i for i in range(ch_groups)], 0
                    )

                ot_map = None
                if self.optimal_transport:
                    ot_map = self.optimal_transport(group, imp, pruning_idxs)

                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist()
                )
                group.ot_map = ot_map
                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self):
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        for group in self.DG.get_all_groups(
            ignored_layers=self.ignored_layers, root_module_types=self.root_module_types
        ):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None:
                    continue
                if ch_groups > 1:
                    imp = imp[: len(imp) // ch_groups]
                global_importance.append((group, ch_groups, imp))

        if len(global_importance) == 0:
            return

        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(self.initial_total_channels *
                                  (1 - target_sparsity))
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)

        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module) // ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices + group_size *
                        i for i in range(ch_groups)], 0
                )
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist()
            )
            if self.DG.check_pruning_group(group):
                yield group
