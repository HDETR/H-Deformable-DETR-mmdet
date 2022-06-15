#!/usr/bin/env python3

import detectron2.engine

import torch.distributed

import datetime


if __name__ == "__main__":
    parser = detectron2.engine.default_argument_parser()
    args = parser.parse_args()
    torch.distributed.init_process_group(
        torch.distributed.Backend.GLOO,
        init_method=args.dist_url,
        timeout=datetime.timedelta(0, 7200),
        world_size=args.num_machines,
        rank=args.machine_rank)
    torch.distributed.barrier()
