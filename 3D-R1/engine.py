import os, sys, time, math, json, importlib
import torch
import datetime
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.ap_calculator import APCalculator
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    init_distributed, 
    is_distributed, 
    is_primary, 
    get_rank,
    barrier,
    all_reduce_average,
    all_gather_dict
)

class Logger:
    def __init__(self, args):
        exp_name = os.path.split(args.checkpoint_dir)[-1]
        self.logger = open(os.path.join(args.checkpoint_dir, f'{exp_name}-logger.log'), 'a')
    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    
    logout = Logger(args)
    
    if is_primary():
        logout(f"call with args: {args}")
        logout(f"{model}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()
    
    max_tolerant_nan = 4
    curr_nan_times = 0
    
    for curr_epoch in range(args.start_epoch, args.max_epoch):
        
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(curr_epoch)
        
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            
            curr_time = time.time()
            
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(net_device)
    
            # Forward pass
            optimizer.zero_grad()
    
            outputs = model(batch_data_label, is_eval=False)
            loss = outputs['loss']
            loss = all_reduce_average(loss)
            
            if not math.isfinite(loss.item()):
                if curr_nan_times < max_tolerant_nan:
                    logout("Loss in not finite. Skip this training step.")
                    curr_nan_times += 1
                    continue
                else:
                    logout("Loss in not finite. Terminate training.")
                    exit(-1)
            curr_nan_times = 0
            
            loss.backward()
            
            # Enhanced gradient clipping with better monitoring
            if args.clip_gradient > 0:
                # Check gradient norms before clipping
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1. / 2)
                
                # Log gradient norm if it's unusually large
                if total_norm > 10.0:
                    logout(f"Warning: Large gradient norm detected: {total_norm:.4f}")
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
                
                # Check gradient norm after clipping
                total_norm_after = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_after += param_norm.item() ** 2
                total_norm_after = total_norm_after ** (1. / 2)
                
                if total_norm_after > args.clip_gradient * 1.1:
                    logout(f"Warning: Gradient clipping may not be effective. Norm after clipping: {total_norm_after:.4f}")
            
            optimizer.step()
    
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
    
            # logging
            if is_primary() and curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
            
            barrier()
            # save ckpt
            if is_primary() and (curr_iter + 1) % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    curr_epoch,
                    args,
                    best_val_metrics,
                    filename=f"checkpoint_{(curr_iter + 1) // 1000}k.pth",
                )
            
            # eval
            if (curr_iter + 1) % args.eval_every_iteration == 0 \
                and (curr_iter + 1) > args.start_eval_after:
                
                eval_metrics = {}
                model.eval()
                for test_loader in dataloaders['test']:
                    task_metrics = test_loader.dataset.eval_func(
                        args,
                        curr_epoch,
                        model,
                        dataset_config,
                        test_loader,
                        logout,
                        curr_train_iter=curr_iter
                    )
                    eval_metrics.update(task_metrics)
                model.train()
                
                if not best_val_metrics or (
                    best_val_metrics[args.criterion] < eval_metrics[args.criterion]
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        model_no_ddp,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    if is_primary():
                        logout(
                            f"Epoch [{curr_epoch}/{args.max_epoch}] "
                            f"saved current best val checkpoint at {filename}; "
                            f"{args.criterion} {eval_metrics[args.criterion]}"
                        )
            # end of an iteration
            
        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
    
    # end of training
    eval_metrics = {}
    model.eval()
    for test_loader in dataloaders['test']:
        task_metrics = test_loader.dataset.eval_func(
            args,
            curr_epoch,
            model,
            dataset_config,
            test_loader,
            logout,
            curr_train_iter=curr_iter
        )
        eval_metrics.update(task_metrics)
    return 
