# Ex-vivo cortical surfaces and three-hinge gyri

Code for reconstructing cortical surfaces from ex-vivo 7T MRI of human hemispheres, and for the analysis of three-hinge gyri (3HG) on those surfaces: their cortical microstructure, its depth profile, their white-matter connectivity, and the replication in vivo.

```
src/
├── run_pipeline.sh    surface reconstruction, one command
├── PIPELINE.md        how to install and run it
├── pipeline/          the scripts it calls
├── export_vtk.py      export to VTK
├── data/              reconstructed surfaces, Subject1-12
└── threehinge/        the 3HG analysis and the figures
```

## Surface reconstruction

```
src/run_pipeline.sh <input.nii.gz> [options]
```

See [PIPELINE.md](PIPELINE.md) for the method, installation, usage and validation.

## Data

`data/` holds the twelve reconstructed right hemispheres as legacy VTK PolyData. They were reconstructed from two public datasets, which hold the original images:

| Specimens | Source | Original data |
|---|---|---|
| Subject1-6 | Zhu et al., *Scientific Data* 2025 | [figshare 28211843](https://springernature.figshare.com/articles/dataset/7_Tesla_multimodal_MRI_dataset_of_ex-vivo_human_brain/28211843) |
| Subject7-12 | Irwin et al., *Brain Communications* 2026 | [Dryad, doi:10.5061/dryad.dbrv15fgk](https://datadryad.org/dataset/doi:10.5061/dryad.dbrv15fgk) |

Subject7-12 are patients 1-6 of the Dryad dataset, in that order. If you use these surfaces, please cite the dataset they come from; the references are at the [end of this page](#citation).

| File | Content |
|---|---|
| `rh.white.vtk`, `rh.pial.vtk`, `rh.mid.vtk` | white, pial and mid surfaces, in scanner coordinates |
| `rh.sphere.vtk` | spherical parameterisation |
| `rh.sphere.reg.vtk` | sphere registered to fsaverage |

Each surface carries `thickness`, `curv`, `sulc`, `area` and `parcel` per vertex. `parcel` is the Desikan–Killiany label, with 0 for the medial wall; the names are in `data/parcel_labels.tsv`.

## Three-hinge analysis

```
threehinge/
├── detect/          gyral crest network and 3HG landmarks (GyralNet)
├── analysis/        sampling of 3HG and their controls, microstructure, depth profiles
├── tractography/    tracking and node-to-node connectivity
├── validation/      controls, null models and sensitivity analyses
├── invivo/          replication in AABC Release 2
├── figures/         one script per figure
└── lib/             shared modules
```

The analysis runs in this order: landmarks are detected on each hemisphere, every modality is sampled onto the surface, 3HG are compared with gyral crowns and sulcal fundi matched to them, tractography is run and the connectivity between landmarks assembled, and the same contrasts are then measured in vivo.

Each figure of the manuscript has its own script in `figures/`:

| Figure | Script | Output |
|---|---|---|
| 1 | `fig_data_overview.py --wide` | `FIG_data_overview_wide.png` |
| 2 | `fig_main1_combined.py --no-f` | `FIG1_signature_ad.png` |
| 3 | `fig_depth_wm.py` | `FIG2_depth_wm.png` |
| 4 | `fig_invivo_folding.py` | `FIG4_invivo_folding.png` |
| 5 | `fig_connectivity_matrix.py` | `FIG5_connectivity.png` |
| Supplementary | `fig_supp_*.py` | `FIGS_*.png` |

The output names follow an older numbering, in which `FIG1_…` is Figure 2 and `FIG2_…` is Figure 3.

```
python src/threehinge/figures/fig_connectivity_matrix.py
```

The location of the data is set in `threehinge/_env.py`, or through the environment:

| Variable | Points at |
|---|---|
| `EXVIVO_ROOT` | the project data: FreeSurfer subjects, landmarks, tractography, results |
| `EXVIVO_AABC` | AABC Release 2 |
| `FREESURFER_HOME` | FreeSurfer 7.x |

## License

The code is released under the MIT [LICENSE](LICENSE). The surfaces in `data/` are derived works and keep the terms of the datasets they come from: Subject1-6 are under CC BY 4.0, following Zhu et al., and Subject7-12 under CC0 1.0, following Irwin et al.

## Citation
If you find our code/data useful in your work, please cite our paper :

```bibtex
@article {chen2026three,
	author = {Chen, Minheng and Dong, Xiuyu and Cheng, Jiale and Hu, Dan and Liu, Tianming and Zhu, Dajiang and Li, Gang},
	title = {Three-hinge gyri link individualized folding topology to cortical and superficial white-matter diffusion organization},
	elocation-id = {2026.10.14.757725},
	year = {2026},
	doi = {10.64898/2026.10.14.757725},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}

```

The surfaces in `data/` are derived from the following datasets. Please cite them as well when using the surfaces.

```bibtex
@article{zhu20257,
  title={7 Tesla multimodal MRI dataset of ex-vivo human brain},
  author={Zhu, Qinfeng and Li, Sihui and Cao, Zuozhen and Shen, Yao and Xu, Haoan and Xu, Guojun and Li, Haotian and Cui, Zhaopeng and Zhu, Keqing and Zhao, Zhiyong and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={845},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{irwin2026clinicoanatomic,
  title={Clinicoanatomic localization of iron-rich gliosis in aphasic presentations of globular glial tauopathy},
  author={Irwin, David J and Emrani, Sheina and Ohm, Daniel T and Trotman, Winifred and Bahena, Alejandra and Teunissen-Bermeo, Eric and Sabatini, Philip and Das, Sandhitsu R and Mizsei, Gabor and Prabhakaran, Karthik and others},
  journal={Brain Communications},
  volume={8},
  number={3},
  pages={fcag169},
  year={2026},
  publisher={Oxford University Press UK}
}
```
