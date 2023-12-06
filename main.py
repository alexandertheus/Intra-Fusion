import copy
import torchvision.transforms as transforms
from utils import evaluate, find_ignored_layers, get_cifar100_data_loader, get_cifar10_data_loader, get_pretrained_model
from prune import prune
from torch_pruning.optimal_transport import OptimalTransport
import torch_pruning as tp
import torch
import json
from parameters import get_parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    dataset = "Cifar10"
    example_inputs = torch.randn(1, 3, 32, 32)
    out_features = 10
    backward_pruning = True
    file_name = f"./models/{args.model_name}.checkpoint"

    config = dict(
        dataset=dataset,
        model=args.model_name,
    )

    loaders = get_cifar10_data_loader()

    model_original, _ = get_pretrained_model(
        config, file_name)

    ot = OptimalTransport(
        p=args.p, target_probability=args.target_prob, source_probability=args.source_prob, target=args.target, gpu_id=args.gpu_id)

    for group_idx in args.group_idxs:
        for sparsity in args.sparsities:
            ########### Pruning model with the conventional pruning procedure ###########
            pruned_model_default = copy.deepcopy(model_original)
            prune(
                pruned_model_default,
                loaders,
                example_inputs,
                out_features,
                args.importance_criteria,
                args.gpu_id,
                sparsity=sparsity,
                backward_pruning=backward_pruning,
                group_idxs=[group_idx],
                dimensionality_preserving=False
            )
            accuracy_pruned_default = evaluate(
                pruned_model_default, loaders, gpu_id=args.gpu_id)

            ########### Pruning model with the Intra-Fusion procedure ###########

            pruned_model_IF = copy.deepcopy(model_original)
            prune(
                pruned_model_IF,
                loaders,
                example_inputs,
                out_features,
                args.importance_criteria,
                args.gpu_id,
                sparsity=sparsity,
                optimal_transport=ot,  # This is the only thing that changed
                backward_pruning=backward_pruning,
                group_idxs=[group_idx],
                dimensionality_preserving=False
            )
            accuracy_pruned_IF = evaluate(
                pruned_model_IF, loaders, gpu_id=args.gpu_id)

            print("-------------------------------------")
            print(f"Group index: {group_idx}. Sparsity: {sparsity}.")
            print(
                f"Default: {accuracy_pruned_default}. Intra-Fusion: {accuracy_pruned_IF}")
