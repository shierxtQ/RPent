# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Fake shared RLDX server: listens on PORT and logs "policy loaded" (``vla_server.sh PORT LOG``).

The log line also records the GPU and CPU thread pools the runner gave it.
"""

import os
import socket
import sys
import time

port, log = int(sys.argv[1]), sys.argv[2]
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen(64)
time.sleep(0.2)
with open(log, "a") as handle:
    pools = " ".join(
        f"{name}={os.environ.get(name)}"
        for name in (
            "RC_GPU",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    )
    handle.write(f"policy loaded {pools}\n")
while True:
    conn, _ = server.accept()
    conn.close()
