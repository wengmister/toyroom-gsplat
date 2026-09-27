# Toyroom reconstruction

Three portrait iPhone videos are sampled at 3 fps into 295 SDR JPEGs at
1080 × 1920. The original videos remain on the Mac at
`/Users/zhengyang/Downloads/toyroom{1,2,3}.MOV`. Their metadata and SHA-256
checksums are recorded in `input-manifest.json`.

The project runs in WSL2 on Alienbot with an RTX 5080. Python environments
are local to this folder. The Windows NVIDIA driver supplies GPU access;
CUDA 12.8 compiler packages are installed inside WSL.

## Frames and cameras

For each clip, extract frames into its own folder beneath `data/images`:

```bash
mkdir -p data/images/toyroom{1,2,3}
ffmpeg -i toyroom1.MOV -vf 'fps=3,scale=-2:1920' -q:v 2 \
  data/images/toyroom1/toyroom1_%05d.jpg
.sfm-venv/bin/python reconstruct.py
```

FFmpeg applies the video's portrait rotation automatically. Repeat extraction
for clips 2 and 3 before reconstructing. COLMAP uses one SIMPLE_RADIAL camera
per clip, GPU SIFT extraction, exhaustive cross-clip matching, and incremental
mapping. Outputs are in `data/sparse`, with registration statistics in
`data/reconstruction-summary.json`. The largest connected model is selected
under `data/aligned` for training.

## Gaussian training

```bash
./train.sh
```

The official gsplat default trainer runs 30,000 steps at the extracted image
resolution. It holds out every eighth registered frame for evaluation and
saves checkpoints and Gaussian PLY files at 7,000 and 30,000 steps under
`results/default`. Automatic trajectory videos
are disabled because interpolating between separate room sweeps can cross
walls. The browser viewer uses port 8080 and stays open after training.

The completed baseline contains 3,201,778 Gaussians. On 31 held-out views it
achieves PSNR 25.70 dB, SSIM 0.861, and LPIPS 0.278. Training and checkpoint
evaluation took about 18 minutes, excluding installation and reconstruction.
The bookshelf, toys, lamp, and TV are recognizable; close curtains, some blank
walls, and occluded furniture remain weak. Comparison images and aggregate
metrics are saved under `results/`.

## View the saved model

While the trainer is running, its viewer is already available. From the Mac:

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8080 wengm@alienbot
```

Open <http://127.0.0.1:8765/>. The tunnel must remain running. To start a viewer
later, after the existing trainer/viewer has exited, run `./view.sh` inside WSL.
It loads the final checkpoint without training. In the viewer, use **Load
Trajectory → inside-room.json → Load → Preview Render** to begin at a captured
camera position. **Exit Render Preview** restores the preceding free camera.
Viewer Res controls interactive rendering resolution, not training resolution.

`results/default/ply/point_cloud_29999.ply` is the final Gaussian export for
compatible splat viewers. `data/sparse-1.ply` is an ordinary sparse point cloud.
They are different representations despite sharing the PLY extension.

## Environment

The upstream source checkout is gsplat v1.5.3 in `vendor/gsplat`; its commit is
recorded in `gsplat-revision.txt`. The training environment uses Python 3.12,
PyTorch 2.9.1+cu128, CUDA toolkit 12.8, and compiled architecture `sm_120`.
`requirements.train.lock` records its complete Python environment, including
pinned Git dependencies and the editable local gsplat checkout.
`requirements.sfm.lock` records the independent GPU COLMAP environment.

The two environments deliberately use different packages named pycolmap.
The reconstruction environment uses the official `pycolmap-cuda12` bindings;
the stable trainer uses the older `rmbrualla/pycolmap` SceneManager reader.
Installing one into the other's environment breaks this integration.

To rebuild the training environment, restore the recorded gsplat revision,
install the pinned CUDA 12.8 PyTorch wheels first, source `cuda-env.sh`, and
install the training lock with build isolation disabled. Source builds also
need Python 3.12 headers, a C++ compiler, Ninja, CUDA NVCC, cudart development
headers, and CCCL. `cuda-env.sh` exposes the NVIDIA headers installed with
PyTorch and limits compilation to two jobs. The Windows GPU driver is used
through WSL; a separate Linux display driver is not installed.

`LEARNING.md` explains every stage, the math behind splatting, and how to
interpret the results.

These splats describe appearance and have arbitrary scale. They are not a
watertight mesh or a metric survey.
