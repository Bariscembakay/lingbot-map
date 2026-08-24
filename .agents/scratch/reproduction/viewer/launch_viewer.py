"""Launch viewer.py's viser server bound to 0.0.0.0 for SSH-tunnel access.

viewer.py's get_host_ip() binds to a specific external IP, not localhost/
0.0.0.0, so an `ssh -L` tunnel has nothing to connect to without this.

Usage (from benchmark/, in the lingbot_map env -- that's the env with viser,
not bench):
    micromamba run -n lingbot_map python \
        ../.agents/scratch/reproduction/viewer/launch_viewer.py <workspace_dir> <port>

e.g.:
    micromamba run -n lingbot_map python \
        ../.agents/scratch/reproduction/viewer/launch_viewer.py \
        /group/compact-3dmem/campaigns/reproduction/viewer_workspace 20540

Then from your own machine:
    ssh -L 20540:localhost:20540 baris_bakay@sof1-h200-0
and open http://localhost:20540
"""
import sys
from pathlib import Path

sys.path.insert(0, '.')
import viewer

viewer.get_host_ip = lambda: "0.0.0.0"
viewer.run_viser(Path(sys.argv[1]), port=int(sys.argv[2]),
                  temporal_subsample=1, spatial_subsample=2, verbose=True)
