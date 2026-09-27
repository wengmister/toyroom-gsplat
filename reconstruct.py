"""Recover cameras and sparse geometry from the three toyroom frame folders."""
import json
from pathlib import Path
import pycolmap

root = Path(__file__).resolve().parent
data = root / 'data'
images = data / 'images'
database = data / 'colmap.db'
sparse = data / 'sparse'
sparse.mkdir(exist_ok=True)
extraction = pycolmap.FeatureExtractionOptions()
extraction.num_threads = 8
extraction.max_image_size = 1920
pycolmap.extract_features(
    database, images, camera_mode=pycolmap.CameraMode.PER_FOLDER,
    reader_options=pycolmap.ImageReaderOptions(camera_model='SIMPLE_RADIAL'),
    extraction_options=extraction, device=pycolmap.Device.cuda,
)
matching = pycolmap.FeatureMatchingOptions()
matching.num_threads = 8
pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cuda)
options = pycolmap.IncrementalPipelineOptions()
options.num_threads = 8
options.random_seed = 42
models = pycolmap.incremental_mapping(database, images, sparse, options=options)
summary = []
for model_id, model in models.items():
    counts = {}
    for image in model.images.values():
        clip = image.name.split('/')[0]
        counts[clip] = counts.get(clip, 0) + 1
    summary.append(dict(model_id=model_id, registered_images=model.num_reg_images(),
                        points3D=model.num_points3D(), mean_reprojection_error=model.compute_mean_reprojection_error(),
                        frames_by_clip=counts))
    model.export_PLY(data / f'sparse-{model_id}.ply')
(data / 'reconstruction-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
if not models:
    raise RuntimeError('No connected reconstruction was recovered.')

best_id = max(models, key=lambda i: models[i].num_reg_images())
aligned = data / "aligned"
aligned.mkdir(exist_ok=True)
for name, target in [("images", "../images"), ("sparse", f"../sparse/{best_id}")]:
    link = aligned / name
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target, target_is_directory=True)
print(f"Selected model {best_id} for training at {aligned}", flush=True)
