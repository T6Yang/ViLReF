import os

from math import ceil
import logging
from pathlib import Path
import json
import time
from time import gmtime, strftime
import importlib.util
import sys

import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch import nn
import torch.multiprocessing as mp

sys.path.append("./ViLReF/")
from clip import load
from clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.feat_extract_img import feat_extract_img


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return args.rank == 0


# used to compare the pytorch version
def torch_version_str_compare_lessequal(version1, version2):
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]
    assert len(v1) == 3, "Cannot parse the version of your installed pytorch! ({})".format(version1)
    assert len(v2) == 3, "Illegal version specification ({}). Should be in 1.X.Y format.".format(version2)
    return sorted([v1, v2])[0] == v1


def debug(rank, world_size=1):
    args = parse_args()

    # Set distributed group
    args.local_device_rank = max(args.local_rank, 0)
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)

    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=rank)
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)

    assert args.precision in ['amp', 'fp16', 'fp32']

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)

    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Build the model
    vision_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v
    model_info['use_flash_attention'] = args.use_flash_attention

    model = CLIP(**model_info)
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
    if args.bert_weight_path is not None:
        assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path,
         use_flash_attention=args.use_flash_attention)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)

    model.cuda(args.local_device_rank)
    if args.precision == "fp16":
        convert_weights(model)

    if args.grad_checkpointing:
        assert not torch_version_str_compare_lessequal(torch.__version__, "1.8.0"), \
            "Currently our grad_checkpointing is not compatible with torch version <= 1.8.0."
        model.set_grad_checkpointing()
        logging.info("Grad-checkpointing activated.")

    if args.use_flash_attention:
        assert importlib.util.find_spec("flash_attn"), "flash_attn is not installed."
        logging.info("Using FlashAttention.")

    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.freeze_vision:
        for k, v in model.visual.named_parameters():
            v.requires_grad = False
        # freeze bn running mean and variance
        if args.vision_model in ['RN50']:
            for m in model.visual.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        logging.info("The visual encoder is freezed during training.")

    # To make compatible with torch version <= 1.8.0, set find_unused_parameters to True
    # In other cases, set find_unused_parameters to False
    find_unused_parameters = torch_version_str_compare_lessequal(torch.__version__, "1.8.0")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_device_rank],
                                                      find_unused_parameters=find_unused_parameters)

    if args.precision == "fp16":
        convert_weights(model)


    # Automatically restore latest checkpoint if exists
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(
                f"=> begin to load checkpoint '{args.resume}'"
            )
            checkpoint = torch.load(args.resume, map_location="cpu")
            sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            # Resize the positional embedding by interpolation, if needed
            resize_pos_embed(sd, model, prefix="module.")
            # Adapt flash attention
            if args.use_flash_attention:
                sd = convert_state_dict(sd)
            # Load the state dict
            model.load_state_dict(sd, False)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.use_visual:
        del model.module.bert
    elif args.use_bert:
        del model.module.visual

    extractor = feat_extract_img(model)

    # load images here :)
    # imgs = [] # tensor: [N, 3, 224, 224]
    imgs = torch.randn([1, 3, 224, 224]).cuda()
    feat = extractor(imgs)


def main():
    mp.spawn(debug, nprocs=1)


if __name__ == "__main__":
    main()
