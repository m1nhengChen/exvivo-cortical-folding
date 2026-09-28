# exvivo-surf

One command from a raw ex-vivo hemisphere to cortical surfaces:

```bash
src/run_pipeline.sh <input.nii.gz> [options]
```

- White, pial and mid surfaces in world coordinates, normals outward
- Spherical parameterisation registered to fsaverage, **no parcellation needed as input**
- DK parcellation, thickness, curvature, sulcal depth, all per-vertex
- Legacy VTK PolyData, loadable in ParaView as-is
- **~10 min** for the surfaces, 1-2 h for registration and parcellation

The white surface is placed on the image itself, mapped into a T1-like contrast, rather than
on a volume synthesised from the segmentation labels, and the pial gets the collapsed sulci
opened first. Both matter on fixed tissue. The method is described in the paper.

## Installation

FreeSurfer 7.x at `$FREESURFER_HOME`, ANTs on `PATH`, and docker with two images:

```
pulks/docker_hippogang_exvivo_segm:v1.4.5   # segmentation (skipped with --seg)
ibeatgroup/ibeat_v2:release210              # sulcus opening, always needed
```

> [!NOTE]
> iBEAT needs a license of your own, requested from the [iBEAT V2.0
> site](https://ibeat.wildapricot.org/) and placed in `$EXVIVO_IB` (default
> `surf/ibeat`).

## Usage

From a raw FLASH volume, fully automatic. Needs a GPU:

```bash
src/run_pipeline.sh hemisphere_FLASH.nii.gz --id 91 --out results/
```

With a segmentation you already have. Any modality, no GPU:

```bash
src/run_pipeline.sh image.nii.gz --seg gm_wm_mask.nii.gz --id 91 --out results/
```

> [!NOTE]
> The segmentation model is FLASH-specific. T2w and other modalities come in through `--seg`
> with a GM/WM mask (GM=1, WM=2).

| Option | Meaning |
|---|---|
| `--seg <mask.nii.gz>` | Use this GM/WM mask instead of segmenting |
| `--id <N>` | Working slot, default 90; must be unused |
| `--out <dir>` | Output directory, default alongside the input |
| `--work <dir>` | Work directory, default `<out>/work` |
| `--threads <T>` | Threads for FreeSurfer stages, default 8 |
| `--keep-work` | Keep intermediates |

## Validation

- **12/12** released subjects built with this chain
- **6/6** agreement with an independent parcellation on the subjects that have one
- End-to-end run reproduces Subject7 **exactly**: 391,981 vertices, 1.963 ± 0.685 mm, 35 parcels

> [!IMPORTANT]
> Parcellation is the part to check on a new subject. The registration route is validated, but
> a new subject has no ground truth of its own. Read the start-to-registration angle printed at
> the end: it must be under **64°**, and normal is **6.9-23.0°**. Well above that, inspect the
> parcellation before using it.
