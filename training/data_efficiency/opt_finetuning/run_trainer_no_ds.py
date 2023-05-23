#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from re import L

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
import deepspeed
import numpy as np
from learning_rates import AnnealingLR


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=None, help="lr schedule warmup ratio."
    )
    parser.add_argument(
        "--decay_style", type=str, default=None, help="lr schedule decay style."
    )
    parser.add_argument(
        "--token_based_lr_decay", action="store_true", help="Use token-based LR decay"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--not_tie_wre", action="store_true", help="tie the last layer and embedding or not."
    )
    parser.add_argument("--random_ltd", action="store_true", help="enable random-ltd or not."
    )
    parser.add_argument("--eval_step", type=int, default=10, help="eval step."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--data_folder", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def print_rank_0(msg, local_rank):
    if local_rank <= 0:
        print(msg)


def set_seeds(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_raw_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_dataset(args, tokenizer):
    raw_datasets = get_raw_dataset(args)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    args.block_size = block_size

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    return train_dataset, eval_dataset


def get_dataloader(train_dataset, eval_dataset, args):
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size
    )
    return train_dataloader, eval_dataloader


def get_model(args):
    if args.model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def evaluation(model, eval_dataset, eval_dataloader, device):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        # batch = tuple(t.to(device) for t in batch)
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.cpu().item())
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(np.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def get_lr_scheduler(optimizer, total_tokens, args):
    if args.token_based_lr_decay:
        lr_scheduler = AnnealingLR(
            optimizer,
            max_lr=args.learning_rate,
            min_lr=0,
            warmup_steps=args.num_warmup_steps,
            decay_tokens=total_tokens,
            decay_style=args.decay_style)
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    return lr_scheduler



def training(model, train_dataloader, train_dataset,
             eval_dataloader, eval_dataset,
             num_train_epochs, device, tokenizer, args):
    start = time.time()
    optimizer = get_optimizer(model, args)
    # Split weights in two groups, one with weight decay and the other not.
    world_size = torch.distributed.get_world_size()
    if args.warmup_ratio is not None:
        args.num_warmup_steps = int(args.max_train_steps*args.warmup_ratio)
    total_tokens = args.max_train_steps*args.per_device_train_batch_size*args.gradient_accumulation_steps*args.block_size*world_size
    lr_scheduler = get_lr_scheduler(optimizer, total_tokens, args)
    # model, optimizer, _, lr_scheduler = deepspeed.initialize(
    #             model=model,
    #             model_parameters=model.parameters(),
    #             # optimizer=optimizer,
    #             args=args,
    #             # lr_scheduler=lr_scheduler,
    #             dist_init_required=True)

    epoch = 0
    global_step = 0
    micro_step = 0
    current_best = float("inf")
    consumed_token = 0
    args.eval_step = max(1, args.max_train_steps // 100)
    while consumed_token < total_tokens:
    # for epoch in range(num_train_epochs):
        if epoch == 0:
            perplexity = evaluation(model, eval_dataset, eval_dataloader, device)
            current_best = min(current_best, perplexity)
            print_rank_0(f"*************************initialization with perplexity {perplexity}***********************************", args.local_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps 
            loss.backward(loss)
            actual_seq_length = args.block_size
            consumed_token += actual_seq_length * args.per_device_train_batch_size * world_size
            optimizer.step()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                global_step += 1
                # Evaluate perplexity on the validation set.
                if global_step%args.eval_step==0 or step == len(train_dataloader)-1:
                    perplexity = evaluation(model, eval_dataset, eval_dataloader, device)
                    current_best = min(current_best, perplexity)
                    log_text = f"At epoch {epoch+1} step {global_step} consumed_token {consumed_token} perplexity {perplexity} current best {current_best}"
                    print_rank_0(log_text, args.local_rank)
            if consumed_token >= total_tokens:
                break
        perplexity = evaluation(model, eval_dataset, eval_dataloader, device)
        current_best = min(current_best, perplexity)
        print_rank_0(f"End of epoch {epoch+1} step {global_step} consumed_token {consumed_token} perplexity {perplexity} current best {current_best}", args.local_rank)
        if consumed_token >= total_tokens:
            break
        epoch += 1
    duration = (time.time() - start) / 3600.0
    print_rank_0(f"End of training epoch {epoch+1} step {global_step} consumed_token {consumed_token} best perplexity {current_best} time {duration} hr", args.local_rank)
    if args.output_dir is not None:
        print_rank_0('saving model ...', args.local_rank)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        if torch.distributed.get_rank() == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save, output_model_file)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)


def main():
    args = parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    if args.seed is not None:
        set_seeds(args.seed)
    torch.distributed.barrier()
    model, tokenizer = get_model(args)
    model.to(device)
    train_dataset, eval_dataset = get_dataset(args, tokenizer)
    train_dataloader, eval_dataloader = get_dataloader(train_dataset, eval_dataset, args)
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Train!
    print_rank_0("***** Running training *****", args.local_rank)
    print_rank_0(f"  Num examples = {len(train_dataset)}", args.local_rank)
    print_rank_0(f"  Num Epochs = {args.num_train_epochs}", args.local_rank)
    print_rank_0(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}", args.local_rank)
    print_rank_0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", args.local_rank)
    print_rank_0(f"  Total optimization steps = {args.max_train_steps}", args.local_rank)
    print_rank_0(f"  Block size (seqlen) = {args.block_size}", args.local_rank)

    num_p = sum([p.numel() for p in model.parameters()])
    print_rank_0('Number of parameters: {}'.format(num_p), args.local_rank)
    training(model, train_dataloader, train_dataset,
             eval_dataloader, eval_dataset,
             args.num_train_epochs, device,
             tokenizer, args)


if __name__ == "__main__":
    main()
