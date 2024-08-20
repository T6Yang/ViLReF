import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="debug",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--use_visual",
        action="store_true",
        default=False,
        help="Only use visual?",
    )
    parser.add_argument(
        "--use_bert",
        action="store_true",
        default=False,
        help="Only use bert?",
    )
    parser.add_argument(
        "--context-length", type=int, default=100,
        help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=2000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
                        default=False,
                        action="store_true",
                        help="Whether to use batch norm sync."
                        )
    parser.add_argument("--use-augment",
                        default=False,
                        action="store_true",
                        help="Whether to use image augment."
                        )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-epoch-frequency", type=int, default=1, help="How often to save checkpoints by epochs."
    )
    parser.add_argument(
        "--save-step-frequency", type=int, default=999999, help="How often to save checkpoints by steps."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        default=True,
        help="If resumed from a checkpoint, whether to reset the optimizer states.",
    )
    parser.add_argument(
        "--reset-data-offset",
        action="store_true",
        default=True,
        help="If resumed from a checkpoint, whether to reset the dataset offset to the beginning.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        # default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--mask-ratio",
        default=0,
        type=float,
        help="Random mask ratio of patches during finetuning. Default to zero which does not mask any patches.",
    )
    parser.add_argument(
        "--clip-weight-path",
        default=None,
        type=str,
        help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained CLIP",
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=False,
        help="Freeze the weight of vision encoder.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--bert-weight-path",
        default=None,
        type=str,
        help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use-flash-attention",
        default=False,
        action="store_true",
        help="Enable flash attention."
    )
    # arguments for distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank."
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed."
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.vision_model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
