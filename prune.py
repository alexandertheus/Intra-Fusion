import torch
import torch_pruning as tp
import numpy as np


def taylor_loss(model, loaders, num_batches, gpu_id):
    loss_func = torch.nn.CrossEntropyLoss()
    if gpu_id != -1:
        model.cuda(gpu_id)

    for batch_idx, (images, labels) in enumerate(loaders["train"]):
        if gpu_id != -1:
            images, labels = images.cuda(), labels.cuda()
        test_output = model(images)

        if batch_idx >= num_batches:
            break

        loss = loss_func(test_output, labels)
        loss.backward()

    model.cpu()


def prune(
    model,
    loaders,
    example_inputs,
    out_features,
    prune_type,
    gpu_id,
    sparsity=0.5,
    optimal_transport=None,
    backward_pruning=True,
    group_idxs=None,
    dimensionality_preserving=False,
    num_batches=5
):
    imp = None

    if prune_type == "random":
        imp = tp.importance.RandomImportance()
    elif prune_type == "l1":
        imp = tp.importance.MagnitudeImportance(1)
    elif prune_type == "l2":
        imp = tp.importance.MagnitudeImportance(2)
    elif prune_type == "l_inf":
        imp = tp.importance.MagnitudeImportance(np.inf)
    elif prune_type == "taylor":
        imp = tp.importance.TaylorImportance()
    elif prune_type == "lamp":
        imp = tp.importance.LAMPImportance()
    elif prune_type == "chip":
        imp = tp.importance.CHIPImportance(feature_maps=None)
    else:
        raise ValueError("Prune type not supported")

    ignored_layers = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    if next(model.parameters()).is_cuda:
        model.to("cpu")

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,  # number of iterations
        ch_sparsity=sparsity,  # channel sparsity
        ignored_layers=ignored_layers,  # ignored_layers will not be pruned
        optimal_transport=optimal_transport,
        backward_pruning=backward_pruning,
        dimensionality_preserving=dimensionality_preserving
    )

    if prune_type == "chip":
        feature_map = pruner.build_feature_map(
            loader=loaders["train"], group_idx=group_idxs[0], num_batches=num_batches)
        imp.set_feature_map(feature_map)

    if prune_type == "taylor":
        taylor_loss(model, loaders, num_batches, gpu_id)

    pruner.step(group_idxs=group_idxs)

    return model
