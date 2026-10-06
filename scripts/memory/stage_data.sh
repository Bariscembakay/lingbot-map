# Sourced by the training job scripts when STAGE=1.
#
# /data/NAME is a squashfs image whose backing blob sits on the zone's CephFS:
# every job re-reads it over the network (gcp-eu1 even reads sof1's CephFS
# cross-site, ~60 MB/s, 5 h for the 944-clip set). Copying each dataset once
# into the node's /scratch makes every later job on that node a local read.
#
# The copy is keyed by the registry blob id, so a dataset re-created with
# --replace never serves a stale copy. flock serialises jobs sharing a node; a
# .complete marker means an interrupted copy is never trusted.

STAGE_ROOT="/scratch/$USER/datacache"

_stage_blob() {
    ls "/data/$1" >/dev/null   # autofs mounts on first access
    mount | awk -v t="/data/$1" '$3 == t {print $1}' | xargs -r basename
}

stage_dataset() {
    local name=$1 blob dst
    blob=$(_stage_blob "$name")
    dst="$STAGE_ROOT/${name}@${blob:-unversioned}"
    mkdir -p "$STAGE_ROOT"
    (
        flock 9
        if [ ! -f "$dst/.complete" ]; then
            echo "[stage] copying /data/$name -> $dst" >&2
            mkdir -p "$dst"
            # One rsync per clip dir: a single stream reads far below what
            # the link sustains with parallel readers.
            (cd "/data/$name" && find . -mindepth 2 -maxdepth 2 -print0 |
                xargs -0 -P 16 -I{} rsync -aR {} "$dst/")
            rsync -a "/data/$name/" "$dst/"   # anything shallower, and a check
            touch "$dst/.complete"
        fi
    ) 9>"$dst.lock"
    echo "$dst"
}

# Map one path or glob from /data/NAME/... to its staged copy.
stage_path() {
    case "$1" in
        /data/*)
            local name rest
            name=$(echo "$1" | cut -d/ -f3)
            rest=${1#/data/$name}
            echo "$(stage_dataset "$name")$rest" ;;
        *) echo "$1" ;;
    esac
}
