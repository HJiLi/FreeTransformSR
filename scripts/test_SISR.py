import argparse
import torch
import torch.nn.functional as F
import os
import logging
import glob
import shutil
from os import path as osp

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.data import create_dataloader, create_dataset
from basicsr.utils import get_root_logger


# ========== Self-ensemble transform functions (unchanged) ==========
def transform(x, mode):
    if mode == 0:
        return x
    elif mode == 1:  # horizontal flip
        return torch.flip(x, [3])
    elif mode == 2:  # vertical flip
        return torch.flip(x, [2])
    elif mode == 3:  # 90-degree rotation
        return torch.rot90(x, 1, [2, 3])
    elif mode == 4:  # 180-degree rotation
        return torch.rot90(x, 2, [2, 3])
    elif mode == 5:  # 270-degree rotation
        return torch.rot90(x, 3, [2, 3])
    elif mode == 6:  # horizontal flip + 90
        return torch.rot90(torch.flip(x, [3]), 1, [2, 3])
    elif mode == 7:  # vertical flip + 90
        return torch.rot90(torch.flip(x, [2]), 1, [2, 3])


def inverse_transform(x, mode):
    if mode == 0:
        return x
    elif mode == 1:
        return torch.flip(x, [3])
    elif mode == 2:
        return torch.flip(x, [2])
    elif mode == 3:
        return torch.rot90(x, 3, [2, 3])
    elif mode == 4:
        return torch.rot90(x, 2, [2, 3])
    elif mode == 5:
        return torch.rot90(x, 1, [2, 3])
    elif mode == 6:
        return torch.flip(torch.rot90(x, 3, [2, 3]), [3])
    elif mode == 7:
        return torch.flip(torch.rot90(x, 3, [2, 3]), [2])


def self_ensemble_predict(model, x):
    outputs = []
    for i in range(8):
        x_t = transform(x, i)
        pred_t = model(x_t)
        pred = inverse_transform(pred_t, i)
        outputs.append(pred)
    return torch.stack(outputs).mean(0)


def iterative_ensemble_refine(model, x, iterations=3):
    outputs = []
    for i in range(8):
        x_t = transform(x, i)
        pred_t = model(x_t)
        pred = inverse_transform(pred_t, i)
        outputs.append(pred)
    outputs = torch.stack(outputs)
    mean_out = outputs.mean(0)
    for _ in range(iterations):
        residuals = outputs - mean_out
        weights = F.softmax(-residuals.abs().mean((1, 2, 3)), dim=0)
        mean_out = (outputs * weights[:, None, None, None]).sum(0)
    return mean_out


class EnsembleModelWrapper(torch.nn.Module):
    def __init__(self, raw_model, use_iter_refine=False):
        super().__init__()
        self.raw_model = raw_model
        self.use_iter_refine = use_iter_refine

    def forward(self, x):
        if self.use_iter_refine:
            return iterative_ensemble_refine(self.raw_model, x)
        else:
            return self_ensemble_predict(self.raw_model, x)


# ========== Cleanup function to remove GT images ==========
def remove_gt_images(save_root):
    """
    Recursively delete all files and folders containing 'gt' in their name.
    This ensures only super-resolved images remain after validation.
    """
    removed = 0
    # Remove files with 'gt' in filename (case insensitive)
    for filepath in glob.glob(osp.join(save_root, '**', '*'), recursive=True):
        if osp.isfile(filepath) and 'gt' in osp.basename(filepath).lower():
            os.remove(filepath)
            removed += 1
    # Remove directories named 'gt' (usually contain GT images)
    for dirpath in glob.glob(osp.join(save_root, '**', 'gt'), recursive=True):
        if osp.isdir(dirpath):
            shutil.rmtree(dirpath, ignore_errors=True)
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model weights.')
    parser.add_argument('--save_img', action='store_true',
                        help='Save output images (GT will be auto-cleaned).')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='Enable self-ensemble test-time augmentation.')
    parser.add_argument('--iter_refine', action='store_true',
                        help='Use iterative error refinement (only with self_ensemble).')
    args = parser.parse_args()

    # Parse config
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    scale = opt['scale']

    logger = get_root_logger(log_level=logging.INFO)

    # Create model and load weights
    model = create_model(opt)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'params_ema' in checkpoint:
        state = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state = checkpoint['params']
    else:
        state = checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_g = model.net_g
    net_g.load_state_dict(state, strict=False)
    net_g.eval()
    net_g.to(device)

    # Apply self-ensemble wrapper if requested
    if args.self_ensemble:
        if args.iter_refine:
            logger.info("Enabled self-ensemble + iterative error refinement")
            wrapped_net = EnsembleModelWrapper(net_g, use_iter_refine=True)
        else:
            logger.info("Enabled self-ensemble test time augmentation")
            wrapped_net = EnsembleModelWrapper(net_g, use_iter_refine=False)
        model.net_g = wrapped_net   # critical: replace the generator

    # Prepare save root directory
    exp_name = opt.get('name', 'test')
    save_root = osp.join('results', exp_name)
    if args.save_img:
        os.makedirs(save_root, exist_ok=True)
        logger.info(f"Images will be saved to: {osp.abspath(save_root)}")

    all_results = {}

    # Loop over validation datasets defined in YAML
    for phase, dataset_opt in opt['datasets'].items():
        if not phase.startswith('val_'):
            continue

        dataset_name = dataset_opt['name']
        logger.info(f'Testing {dataset_name}')

        try:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=0
            )
        except Exception as e:
            logger.warning(f'Failed to create dataset {dataset_name}: {e}')
            continue

        # Standard BasicSR validation (may save GT images if model defines them)
        metrics = model.validation(
            val_loader,
            current_iter=0,
            tb_logger=None,
            save_img=args.save_img
        )

        # Auto‑clean any GT images that were saved
        if args.save_img:
            removed = remove_gt_images(save_root)
            if removed > 0:
                logger.info(f'Cleaned {removed} GT-related files/folders from {save_root}')

        # Collect metrics
        if isinstance(metrics, dict):
            psnr_val = metrics.get('psnr')
            ssim_val = metrics.get('ssim')
            if psnr_val is not None and ssim_val is not None:
                all_results[dataset_name] = {'psnr': psnr_val, 'ssim': ssim_val}
                logger.info(f'{dataset_name}: PSNR = {psnr_val:.4f} dB, SSIM = {ssim_val:.4f}')
            else:
                logger.warning(f'No PSNR/SSIM in metrics for {dataset_name}')
        else:
            logger.warning(f'validation() returned non-dict: {type(metrics)}')

    # Print summary table
    if all_results:
        print('\n' + '=' * 60)
        print(f'{"Dataset":<20} {"PSNR (dB)":<15} {"SSIM":<15}')
        print('-' * 60)
        for name, m in all_results.items():
            print(f'{name:<20} {m["psnr"]:.4f}          {m["ssim"]:.4f}')
        print('=' * 60)

    if args.save_img:
        print(f'\nAll super-resolved images saved to: {osp.abspath(save_root)}')


if __name__ == '__main__':
    main()