import argparse
import os
import torch
import subprocess as sp
import time


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config_file", help="config file for train and test", type=str, required=True,
    )
    parser.add_argument(
        "--dataset_names",
        help="used dataset names, split with ','",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--blob_root", help="path to blob root", type=str, required=True,
    )
    parser.add_argument(
        "--zip_filename", help="zip codebase filename", type=str, required=True,
    )
    parser.add_argument(
        "--output_path", help="output path on blob", type=str, required=True,
    )
    parser.add_argument(
        "--unparsed", help="unparsed", default="", type=str,
    )
    parser.add_argument(
        "--working_dir", required=True, default="", type=str,
    )
    args = parser.parse_args()
    extra_args = args.unparsed
    return args, extra_args


def build_repo(args):

    cmd = f"pip install -r {args.working_dir}/requirements.txt"
    print(cmd)
    os.system(cmd)

    sh_path = os.path.join(args.working_dir, "models", "ops")

    cmd = f"sudo sh make.sh"
    print(cmd)
    sp.run(cmd, shell=True, cwd=sh_path, check=True)


def prepare_datasets(args):
    dataset_path = os.path.join(args.working_dir, "data")
    dataset_names = args.dataset_names.split(",")
    azcopy_dataset_names = ["ADEChallengeData2016", "coco", "cityscapes"]
    for dataset_name in dataset_names:
        cmd = f"ln -s {args.blob_root}/dataset/{dataset_name} {dataset_path}/"
        print(cmd)
        os.system(cmd)
        cmd = f"ls {dataset_path}"
        print(cmd)
        os.system(cmd)


def unzip_codebase(args):
    codebase_filepath = os.path.join(args.output_path, args.zip_filename)
    cmd = f"unzip {codebase_filepath} -d {args.working_dir} >/dev/null\n"
    print(cmd)
    os.system(cmd)
    cmd = f"ls {args.working_dir}"
    print(cmd)
    os.system(cmd)


def install_apps():
    cmd = "sudo rm /etc/apt/sources.list.d/cuda.list \n "
    cmd += "sudo rm /etc/apt/sources.list.d/nvidia-ml.list \n"
    cmd += "apt-key del 7fa2af80 \n"
    cmd += "apt-get update && apt-get install -y --no-install-recommends wget \n"
    cmd += "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \n"
    cmd += "sudo dpkg -i cuda-keyring_1.0-1_all.deb \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    cmd = (
        "apt-get update && apt-get install -y --no-install-recommends sudo >/dev/null\n"
    )
    cmd += "sudo apt -y --fix-broken install && sudo apt-get update && sudo apt-get install -y --no-install-recommends zip unzip expect vim-gtk libssl-dev pigz time python3-dev >/dev/null"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    os.environ["PATH"] += ":{}/.local/bin".format(os.environ["HOME"])
    cmd = "export PATH=$HOME/.local/bin:$PATH"
    print(cmd)
    os.system(cmd)

    cmd = "pip install torch==1.10.1 torchvision==0.11.2 >/dev/null"
    print(cmd)
    os.system(cmd)

    cmd = "python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html"
    print(cmd)
    os.system(cmd)

    cmd = "pip install opencv-python einops >/dev/null"
    print(cmd)
    os.system(cmd)

    cmd = "pip install git+https://github.com/cocodataset/panopticapi.git"
    print(cmd)
    os.system(cmd)

    cmd = "pip install git+https://github.com/mcordts/cityscapesScripts.git"
    print(cmd)
    os.system(cmd)

    cmd = "pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html"
    print(cmd)
    os.system(cmd)

    cmd = "pip install openmim"
    print(cmd)
    os.system(cmd)

    cmd = "mim install mmdet"
    print(cmd)
    os.system(cmd)

    cmd = "pip install timm wandb"
    print(cmd)
    os.system(cmd)


def barrier(args, machine_rank, num_machines, dist_url):
    # sync
    print(f"Node {machine_rank} enters barrier.")

    cmd = f"export LC_ALL=C.UTF-8"
    print(cmd)
    os.system(cmd)

    cmd = f"export LANG=C.UTF-8"
    print(cmd)
    os.system(cmd)
    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"

    os.environ["OPENBLAS_NUM_THREADS"] = "12"

    install_apps()
    unzip_codebase(args)

    build_repo(args)
    prepare_datasets(args)

    cmd = "export GLOO_SOCKET_IFNAME=eth0\n"
    cmd += f"python run_on_aml/barrier.py --dist-url {dist_url} --machine-rank {machine_rank} --num-machines {num_machines}"
    print(cmd)
    sp.run(cmd, shell=True, cwd=args.working_dir, check=True)
    print(f"Node {machine_rank} exits barrier.")


def main():
    # install_apps()
    args, extra_args = parse_args()
    args.working_dir = "/mnt"
    cmd = f"sudo mkdir -p {args.working_dir} \n"
    cmd += f"sudo chmod -R 777 {args.working_dir} \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    args.log_dir = os.path.join(args.working_dir, "temp_log")
    cmd = f"sudo mkdir -p {args.log_dir} \n"
    cmd += f"sudo chmod -R 777 {args.log_dir} \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    args.output_path = os.path.join("/blob", args.output_path)
    cmd = f"sudo ln -s {args.blob_root} /blob \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

    cmd = f"sudo ln -s {args.blob_root}/jiading/pretrained_backbone {args.working_dir}/pretrained_backbone \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

    num_machines = int(os.getenv("OMPI_COMM_WORLD_SIZE", default="1"))
    machine_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", default="0"))
    cmd = "env | grep NODE\n env | grep OMPI \n env | grep MASTER\n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)
    print(f"num_machines={num_machines}, machine_rank={machine_rank}")
    gpus = torch.cuda.device_count()

    if num_machines > 1:
        if "MASTER_IP" not in os.environ.keys():
            # OCRA or OCR2
            AZ_BATCH_MASTER_NODE = os.environ["AZ_BATCH_MASTER_NODE"]
            dist_url = f"tcp://{AZ_BATCH_MASTER_NODE}"
        else:
            # k8s compute target
            master_ip = os.environ["MASTER_IP"]
            master_port = os.environ["MASTER_PORT"]
            dist_url = f"tcp://{master_ip}:{master_port}"
        # sync before training
        barrier(args, machine_rank, num_machines, dist_url)
        # start training
        cmd = "export GLOO_SOCKET_IFNAME=eth0\n"

        cmd += (
            f"python train_net.py --resume "
            f"--dist-url {dist_url} "
            f"--machine-rank {machine_rank} "
            f"--num-machines {num_machines} "
            f"--config-file {args.config_file} "
            f"--num-gpus {gpus} "
            f"LOG_TEMP_OUTPUT {args.log_dir} "
            f"OUTPUT_DIR {args.output_path} {extra_args} "
        )
        print(cmd)
        sp.run(cmd, shell=True, check=True, cwd=args.working_dir)
    else:
        cmd = f"export LC_ALL=C.UTF-8"
        print(cmd)
        os.system(cmd)

        cmd = f"export LANG=C.UTF-8"
        print(cmd)
        os.system(cmd)
        os.environ["LANG"] = "C.UTF-8"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["OPENBLAS_NUM_THREADS"] = "12"
        install_apps()
        unzip_codebase(args)

        build_repo(args)
        prepare_datasets(args)
        if "checkpoint.pth" in os.listdir(args.output_path):
            cmd = (
                f"GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  "
                f"{args.config_file} "
                f"--output_dir {args.output_path} --resume {args.output_path}/checkpoint.pth 2>&1 |tee {args.output_path}/azure_log.txt"
            )
        else:
            cmd = (
                f"GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  "
                f"{args.config_file} "
                f"--output_dir {args.output_path} --resume  /blob/jiading/outputs-aml/Deformable-DETR-gt-repeat-in-branch-tricks-V3-6-14/two_stage/deformable-detr-baseline/24eps/24eps_1024dim_baseline_MS_LFW_dp0_r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/checkpoint0009.pth  2>&1 |tee {args.output_path}/azure_log.txt"
            )
        print(cmd)
        sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

    # copy the log frm working_dir to blob
    """cmd = f"cp -r {args.log_dir} {args.output_path}"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)"""


if __name__ == "__main__":
    main()
