# WS1 Published Reference Numbers

All values below are transcribed from the primary papers. They are references,
not results produced by this repository, and must not be compared directly with
WS1 results when datasets, inputs, offset protocols, or metric definitions differ.

## InSynFormer (AAAI 2024)

Paper: Yuxuan Liu, Haizhou Ai, Junliang Xing, Xuri Li, Xiaoyi Wang, and Pin Tao,
“Advancing Video Synchronization with Fractional Frame Analysis: Introducing a
Novel Dataset and Model,” AAAI 2024.

Table 1, IFID test set:

| Method | Accex@1 | Accex@3 | Accin@1 | Accin@3 | Accin@5 | Frm.err |
|---|---:|---:|---:|---:|---:|---:|
| SynNet | 60.41% | 91.37% | 31.78% | 70.12% | 85.62% | 1.26 |
| SeSyn-Net | 79.93% | 92.44% | — | — | — | 0.87 |
| CNNSiamese | 36.42% | 76.34% | — | — | — | 2.04 |
| InSynFormer | **80.86%** | **94.35%** | **61.30%** | **90.69%** | **95.49%** | **0.83** |

Primary sources: [AAAI article page](https://ojs.aaai.org/index.php/AAAI/article/view/28174)
and [paper PDF](https://ojs.aaai.org/index.php/AAAI/article/view/28174/28346).

## SyncTrack4D (arXiv 2025)

Paper: Yonghan Lee, Tsung-Wei Huang, Shiv Gehlot, Jaehoon Choi, Guan-Ming Su,
and Dinesh Manocha, “SyncTrack4D: Cross-Video Motion Alignment and Video
Synchronization with Multi-Video 4D Gaussian Splatting,” arXiv:2512.04315,
2025.

Table 1, two-view initialization error in frames:

| Scene | 0 ≤ \|Δt\| ≤ 10 Data | 0 ≤ \|Δt\| ≤ 10 Ours (Init) | 10 ≤ \|Δt\| ≤ 30 Data | 10 ≤ \|Δt\| ≤ 30 Ours (Init) | 30 ≤ \|Δt\| ≤ 50 Data | 30 ≤ \|Δt\| ≤ 50 Ours (Init) |
|---|---:|---:|---:|---:|---:|---:|
| boxes | 7.00 | 1.97 | 24.00 | 2.37 | 37.14 | 2.55 |
| juggle | 7.00 | 4.72 | 24.00 | 2.51 | 37.14 | 22.62 |
| softball | 7.00 | 1.58 | 24.00 | 1.37 | 37.14 | 1.45 |
| tennis | 7.00 | 1.90 | 24.00 | 3.72 | 37.14 | 3.86 |
| basketball | 7.00 | 6.44 | 24.00 | 8.57 | 37.14 | 9.87 |
| football | 7.00 | 7.09 | 24.00 | 8.60 | 37.14 | 8.84 |

Table 2, many-view average temporal error in frames:

| Scene | Init. | SyncNeRF | Ours (DTW Init) | Ours (DTW + Refine) |
|---|---:|---:|---:|---:|
| boxes | 13.355 | 13.387 | 2.379 | 0.205 |
| juggle | 13.355 | 13.414 | 6.414 | 0.624 |
| softball | 13.355 | 13.443 | 1.379 | 0.146 |
| tennis | 13.355 | 13.375 | 3.724 | 0.187 |
| basketball | 13.355 | 13.416 | 9.689 | 0.260 |
| football | 13.355 | 13.405 | 9.960 | 0.138 |
| **Average** | **13.355** | **13.405** | **5.590** | **0.260** |

Primary source: [arXiv HTML, Tables 1–2](https://arxiv.org/html/2512.04315).

```bibtex
@inproceedings{liu2024insynformer,
  title={Advancing Video Synchronization with Fractional Frame Analysis:
         Introducing a Novel Dataset and Model},
  author={Liu, Yuxuan and Ai, Haizhou and Xing, Junliang and Li, Xuri and
          Wang, Xiaoyi and Tao, Pin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
  doi={10.1609/aaai.v38i4.28174}
}

@article{lee2025synctrack4d,
  title={SyncTrack4D: Cross-Video Motion Alignment and Video Synchronization
         with Multi-Video 4D Gaussian Splatting},
  author={Lee, Yonghan and Huang, Tsung-Wei and Gehlot, Shiv and Choi, Jaehoon
          and Su, Guan-Ming and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2512.04315},
  year={2025}
}
```
