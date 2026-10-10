#!/usr/bin/env bash
#SBATCH --job-name=imagenet-download
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=imagenet-download-%j.out

# The partition also accepts CPU jobs. No GPU is requested for file transfer.
set -euo pipefail
repo_root=${SLURM_SUBMIT_DIR:?Submit from the repository root}
data_root=${1:?Usage: sbatch scripts/hendrix/imagenet_download.sbatch /absolute/data/path}
[[ "$data_root" == /* ]] || { echo 'Use an absolute data path.' >&2; exit 2; }
cd -- "$repo_root"
if [[ -n ${MEANFLOW_PYTHON_MODULE:-} ]]; then
  module load "$MEANFLOW_PYTHON_MODULE"
fi
venv_dir=${MEANFLOW_VENV:-"$repo_root/.venv-hendrix"}
[[ -x "$venv_dir/bin/python" ]] || { echo 'The Hendrix environment is missing.' >&2; exit 1; }
mkdir -p -- "$data_root"
exec 9>"$data_root/.imagenet-download.lock"
flock -n 9 || { echo 'Another preparation job is using this data directory.' >&2; exit 1; }
archive="$data_root/ILSVRC2012_img_train.tar"
url='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'
hostname
df -h -- "$data_root"
if command -v aria2c >/dev/null; then
  aria2c -c -x8 -s8 --file-allocation=none --auto-file-renaming=false \
    --dir="$data_root" --out=ILSVRC2012_img_train.tar "$url"
else
  [[ ! -f "$archive.aria2" ]] || {
    echo 'This download has aria2 state. Load aria2c and resume with it, not wget.' >&2
    exit 1
  }
  wget -c -O "$archive" "$url"
fi
# Verify the official training archive before extracting any images.
actual_md5=$(md5sum -- "$archive")
[[ ${actual_md5%% *} == 1d675b47d978889d74fa0da5fadfb00e ]] || {
  echo 'ImageNet archive checksum mismatch. Inspect the download before retrying.' >&2
  exit 1
}
train_root="$data_root/imagenet/train"
markers="$data_root/.imagenet-unpack"
mkdir -p -- "$train_root" "$markers"
if [[ ! -f "$markers/outer.done" ]]; then
  tar -xf "$archive" -C "$train_root"
  touch "$markers/outer.done"
fi
shopt -s nullglob
for class_archive in "$train_root"/*.tar; do
  class_name=${class_archive##*/}
  class_name=${class_name%.tar}
  [[ "$class_name" =~ ^n[0-9]{8}$ ]] || { echo "Unexpected class: $class_name" >&2; exit 1; }
  if [[ ! -f "$markers/$class_name.done" ]]; then
    mkdir -p -- "$train_root/$class_name"
    tar -xf "$class_archive" -C "$train_root/$class_name"
    touch "$markers/$class_name.done"
    echo "Extracted $class_name"
  fi
  # Only the intermediate class tar is removed, after successful extraction.
  rm -- "$class_archive"
done
"$venv_dir/bin/python" scripts/hendrix/check_imagenet.py raw "$data_root/imagenet"
echo "Raw ImageNet is ready: $data_root/imagenet"
