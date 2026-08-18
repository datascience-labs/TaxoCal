import argparse
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='TaxoCal Stage-2 (resume-to-job matching) parameters'
    )
    # --- data ---
    parser.add_argument('--path', default='data/dataset.pkl', type=str, help="data path")
    parser.add_argument('--hpath', default='data/h_dataset.pkl', type=str, help="hierarchy data path")

    # --- optimisation (unchanged defaults: these are the settings reported in the paper) ---
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--pos_dim', default=64, type=int, help='positional embedding dimenssion')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--aug_batch', default=256, type=int, help='augmentation batch size')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--num_neg_samples', default=127, type=int, help='num_neg_samples')
    parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.2, type=float, help='rate for edge dropout')
    parser.add_argument('--lambda1', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--temp', default=.25, type=float, help='temperature in cl loss')
    parser.add_argument('--activation', default=0.1, type=float, help='LeakyReLU Negative Slope')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')

    # --- data / checkpoint I/O ---
    # All default to None so that the historical behaviour is reproduced exactly:
    #   ckpt_dir        -> <repo>/checkpoints/
    #   pretrained_ckpt -> <ckpt_dir>/{epoch}_checkpoints.pth
    #   inference_path  -> <repo>/data/inference.pkl
    parser.add_argument('--inference_path', default=None, type=str,
                        help='matching dataset produced by stage-1 preprocessing '
                             '(default: <repo>/data/inference.pkl)')
    parser.add_argument('--ckpt_dir', default=None, type=str,
                        help='directory for reading/writing checkpoints (default: <repo>/checkpoints)')
    parser.add_argument('--pretrained_ckpt', default=None, type=str,
                        help='stage-1b checkpoint holding j_embeddings/s_embeddings '
                             '(default: <ckpt_dir>/{epoch}_checkpoints.pth)')

    # --- reproducibility ---
    # Default None keeps the original, unseeded behaviour bit-for-bit. Supplying a seed
    # makes the 50/10/40 random_split deterministic, so a run can be repeated exactly.
    # operate on the same test split.
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed; omit for the original unseeded behaviour')
    return parser.parse_args()


def resolve(path, root=ROOT):
    """Resolve a possibly-relative path against the repository root.

    Absolute paths are returned unchanged, so ``--path /data/foo.pkl`` still works.
    This makes every entry point runnable from any working directory.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(root, path)


args = parse_args()
