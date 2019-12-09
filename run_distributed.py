# references
# https://pytorch.org/tutorials/intermediate/dist_tuto.html
# https://github.com/pytorch/examples/tree/master/imagenet

import argparse
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing
import torch.multiprocessing as mp

"""Blocking point-to-point communication."""
def run(rank, world_size):
    tensor = torch.zeros(1)
    if rank == 0:
        # receive from all other processes
        for other_rank in range(1, world_size):
            req=dist.recv(tensor=tensor, src=other_rank)
            print('received from other_rank %s\n%s' %(other_rank, tensor))
    else:
        # process and return tensor
        a=0
        for i in range(10**5):
            a+=1
        req=dist.send(tensor=tensor, dst=0)

    print('Rank ', rank, ' has data ', tensor[0])

def init_process(process_rank, core_count, args, world_size):
    """ Initialize the distributed environment. """
    #dist.init_process_group(backend, rank=rank, world_size=size)
    global_process_rank = (args.node_rank*core_count) + process_rank    # assuming same core number for each node
    dist.init_process_group(
        backend='gloo',
        init_method=args.dist_url,
        world_size=world_size,
        rank=global_process_rank
    )
    #fn(rank, size)
    run( global_process_rank, world_size )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Pomdp Solvers')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url of the host')
    parser.add_argument('--node_rank', default=-1, type=int, help='rank of this node')
    parser.add_argument('--total_node_count', default=-1, type=int, help='number of nodes for distributed training')
    args = parser.parse_args()

    core_count = multiprocessing.cpu_count()
    world_size = core_count * args.total_node_count
    mp.spawn(init_process, nprocs=core_count, args=(core_count, args, world_size))

# commands on a two node system:
# python3 run_distributed.py --dist_url 'tcp://localhost:11312' --node_rank 0 --total_node_count 2
# python3 run_distributed.py --dist_url 'tcp://localhost:11312' --node_rank 1 --total_node_count 2
