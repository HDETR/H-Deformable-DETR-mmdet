import argparse
import subprocess as sp
import time


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--blob_root",
        help="path to blob root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sleep_for_debug",
        help="sleep hours for debug",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    return args


def install_apps():
    cmd = "apt-get update && apt-get install -y --no-install-recommends sudo >/dev/null\n"
    cmd += "sudo apt -y --fix-broken install && sudo apt-get update && sudo apt-get install -y --no-install-recommends zip unzip expect vim-gtk libssl-dev pigz time python3-dev >/dev/null"
    print(cmd)
    sp.run(cmd, shell=True, check=True)


def main():
    install_apps()
    args = parse_args()
    working_dir = "/mnt/codes"
    cmd = f"sudo mkdir -p {working_dir} \n"
    cmd += f"sudo chmod -R 777 {working_dir} \n"
    cmd += f"sudo ln -s {args.blob_root} /blob \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)
    if args.sleep_for_debug > 0:
        time.sleep(3600 * args.sleep_for_debug)


if __name__ == '__main__':
    main()