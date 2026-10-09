"""An isolated local Mooncake storage segment for the benchmark."""

import argparse
import time

from mooncake.store import MooncakeDistributedStore

p = argparse.ArgumentParser()
p.add_argument("--master", required=True)
a = p.parse_args()
store = MooncakeDistributedStore()
assert (
    store.setup("127.0.0.1", "P2PHANDSHAKE", 96 << 30, 64 << 20, "tcp", "", a.master)
    == 0
)
print("server up", flush=True)
while True:
    time.sleep(3600)
