import setproctitle
import torch
import argparse
import time

setproctitle.setproctitle("测试性能，请勿占用")

parser = argparse.ArgumentParser()
parser.add_argument("--dev", type=int, default=0)
args = parser.parse_args()

print(f"Taking Device: {args.dev}")
while True:
    A = torch.rand([1, 1], dtype=torch.float32).cuda(args.dev)
    time.sleep(1000)