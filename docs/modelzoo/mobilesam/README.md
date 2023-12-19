## Introduction

<a href="https://github.com/ChaoningZhang/MobileSAM">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/mobilesam/mobilesam.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2306.14289.pdf">MobileSAM (ArXiv'2023)</a></summary>

```latex
@article{mobile_sam,
    title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
    author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
    journal={arXiv preprint arXiv:2306.14289},
    year={2023}
}
```

</details>


## Inference with MobileSAM

The usage of MobileSAM in sssegmenation is exactly the same as SAM by replacing

- `SAM`: `MobileSAM`,
- `SAMPredictor`: `MobileSAMPredictor`,
- `SAMAutomaticMaskGenerator`: `MobileSAMAutomaticMaskGenerator`.

Specifically, you can import the three classes by

```python
from ssseg.modules.models.segmentors.mobilesam import MobileSAM
from ssseg.modules.models.segmentors.mobilesam import MobileSAMPredictor
from ssseg.modules.models.segmentors.mobilesam import MobileSAMAutomaticMaskGenerator

# predictor only could be MobileSAMPredictor(use_default_sam_t_5m=True, device='cuda')
predictor = MobileSAMPredictor(use_default_sam_t_5m=True, device='cuda')

# mask_generator only could be MobileSAMAutomaticMaskGenerator(use_default_sam_t_5m=True, device='cuda')
mask_generator = MobileSAMAutomaticMaskGenerator(use_default_sam_t_5m=True, device='cuda')
```

By the way, you can refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) to learn about how to use SAM with sssegmenation.
Also, you can refer to [MobileSAM Official Repo](https://github.com/ChaoningZhang/MobileSAM) to compare our implemented MobileSAM with official version.