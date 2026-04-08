
import argparse
from .trainer import Trainer
import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--config',type=str,required=True)
    p.add_argument('--ddp', action='store_true', help='Active DistributedDataParallel')
    
    default_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    p.add_argument('--gpus', type=int, default=default_gpus, help="Nb de GPUs/processus")
    
    return p.parse_args()


def _worker(rank, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=args.gpus)
    try:
        trainer = Trainer(args, ddp=True, rank=rank, world_size=args.gpus)
        trainer.fit()
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            print(f"[R{rank}] destroy_process_group()", flush=True)
            dist.destroy_process_group()


def train():
    args= parse_args()
    if args.ddp and args.gpus > 1:
        print(f"DDP ON : {args.gpus} GPU(s)\n")
        mp.spawn(_worker,args=(args,),nprocs=args.gpus,join=True)
    else:
        print(f"DDP OFF\n")
        trainer = Trainer(args)
        trainer.fit()


if __name__ == '__main__':
    train()
