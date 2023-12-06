import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default='vgg11_bn',
                        help='Type of neural network model (vgg11_bn|resnet18)')
    parser.add_argument('--group_idxs', nargs='+', type=int,
                        default=[1, 2, 3], help='Groups to prune.')
    parser.add_argument('--sparsities', nargs='+', type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7],
                        help='Target sparsities. Target sparsity of 0.5 means 50 percent of neural pairings are removed.')
    parser.add_argument('--source_prob', default='uniform', type=str,
                        help='Source probability of the OT problem (uniform|importance).')
    parser.add_argument('--target_prob', default='uniform', type=str,
                        help='Target probability of the OT problem (uniform|importance).')
    parser.add_argument('--p', type=int, default=1,
                        help='p value of p-norm used to calculate neural similarity.')
    parser.add_argument('--target', type=str,
                        default='most_important', help='Target used for the OT problem (most_important|cluster_centroid).')
    parser.add_argument('--importance_criteria', type=str, default="l1",
                        help='Importance criterion to quantify neuron importance (l1|taylor|lamp|chip). Warning: CHIP is slow.')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU id of the GPU used. For CPU use "-1"')

    return parser
