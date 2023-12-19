## Introduction

<a href="https://github.com/chongzhou96/EdgeSAM">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/edgesam/edgesam.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2312.06660.pdf">EdgeSAM (ArXiv'2023)</a></summary>

```latex
@article{zhou2023edgesam,
    title={EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM},
    author={Zhou, Chong and Li, Xiangtai and Loy, Chen Change and Dai, Bo},
    journal={arXiv preprint arXiv:2312.06660},
    year={2023}
}
```

</details>


## Inference with EdgeSAM

The usage of EdgeSAM in sssegmenation is exactly the same as SAM by replacing

- `SAM`: `EdgeSAM`,
- `SAMPredictor`: `EdgeSAMPredictor`,
- `SAMAutomaticMaskGenerator`: `EdgeSAMAutomaticMaskGenerator`.

Specifically, you can import the three classes by

```python
from ssseg.modules.models.segmentors.edgesam import EdgeSAM
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.edgesam import EdgeSAMAutomaticMaskGenerator

# predictor could be EdgeSAMPredictor(use_default_edgesam=True, device='cuda') or EdgeSAMPredictor(use_default_edgesam_3x=True, device='cuda')
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cuda')

# mask_generator could be EdgeSAMAutomaticMaskGenerator(use_default_edgesam=True, device='cuda') or EdgeSAMAutomaticMaskGenerator(use_default_edgesam_3x=True, device='cuda')
mask_generator = EdgeSAMAutomaticMaskGenerator(use_default_edgesam=True, device='cuda')
```

By the way, you can refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) to learn about how to use SAM with sssegmenation.
Also, you can refer to [EdgeSAM Official Repo](https://github.com/chongzhou96/EdgeSAM) to compare our implemented EdgeSAM with official version.