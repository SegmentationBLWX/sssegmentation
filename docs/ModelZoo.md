# Model Zoo


## Common Settings

- We use distributed training with 8 GPUs by default.
- Our ResNet style backbone are based on ResNetV1c variant, where the 7x7 conv in the input stem is replaced with three 3x3 convs.
- There are two inference modes in this framework.
	- slide mode: In this mode, multiple patches will be cropped from input image, passed into network individually. The overlapping area will be merged by average.
	- whole mode: In this mode, the whole imaged will be passed into network directly.


## Supported Backbones

**1.UNet**

- Related Paper: [click](https://arxiv.org/pdf/1505.04597.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/unet).

**2.Twins**

- Related Paper: [click](https://arxiv.org/pdf/2104.13840.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/twins).

**3.CGNet**

- Related Paper: [click](https://arxiv.org/pdf/1811.08201.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/cgnet).

**4.HRNet**

- Related Paper: [click](https://arxiv.org/pdf/1908.07919.pdf).

**5.ERFNet**

- Related Paper: [click](https://ieeexplore.ieee.org/document/8063438),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/erfnet).

**6.ResNet**

- Related Paper: [click](https://arxiv.org/pdf/1512.03385.pdf).

**7.ResNeSt**

- Related Paper: [click](https://arxiv.org/pdf/2004.08955.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/resnest).

**8.FastSCNN**

- Related Paper: [click](https://arxiv.org/pdf/1902.04502.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/fastscnn).

**9.BiSeNetV1**

- Related Paper: [click](https://arxiv.org/pdf/1808.00897.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/bisenetv1).

**10.BiSeNetV2**

- Related Paper: [click](https://arxiv.org/pdf/2004.02147.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/bisenetv2).

**11.MobileNetV2**

- Related Paper: [click](https://arxiv.org/pdf/1801.04381.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/mobilenet).

**12.MobileNetV3**

- Related Paper: [click](https://arxiv.org/pdf/1905.02244.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/mobilenet).

**13.SwinTransformer**

- Related Paper: [click](https://arxiv.org/pdf/2103.14030.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/swin).

**14.VisionTransformer**

- Related Paper: [click](https://arxiv.org/pdf/2010.11929.pdf).

**15.ConvNeXt**

- Related Paper: [click](https://arxiv.org/pdf/2201.03545.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/convnext).


## Supported Segmentors

**1.FCN**

- Related Paper: [click](https://arxiv.org/pdf/1411.4038.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/fcn).

**2.CE2P**

- Related Paper: [click](https://arxiv.org/pdf/1809.05996.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/ce2p).

**3.SETR**

- Related Paper: [click](https://arxiv.org/pdf/2012.15840.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/setr).

**4.ISNet**

- Related Paper: [click](https://arxiv.org/pdf/2108.12382.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/isnet).

**5.ICNet**

- Related Paper: [click](https://arxiv.org/pdf/1704.08545.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/icnet).

**6.CCNet**

- Related Paper: [click](https://arxiv.org/pdf/1811.11721.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/ccnet).

**7.DANet**

- Related Paper: [click](https://arxiv.org/pdf/1809.02983.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/danet).

**8.GCNet**

- Related Paper: [click](https://arxiv.org/pdf/1904.11492.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/gcnet).

**9.DMNet**

- Related Paper: [click](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/dmnet).

**10.ISANet**

- Related Paper: [click](https://arxiv.org/pdf/1907.12273.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/isanet).

**11.EncNet**

- Related Paper: [click](https://arxiv.org/pdf/1803.08904.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/encnet).

**12.OCRNet**

- Related Paper: [click](https://arxiv.org/pdf/1909.11065.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/ocrnet).

**13.DNLNet**

- Related Paper: [click](https://arxiv.org/pdf/2006.06668.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/dnlnet).

**14.ANNNet**

- Related Paper: [click](https://arxiv.org/pdf/1908.07678.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/annnet).

**15.EMANet**

- Related Paper: [click](https://arxiv.org/pdf/1907.13426.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/emanet).

**16.PSPNet**

- Related Paper: [click](https://arxiv.org/pdf/1612.01105.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/pspnet).

**17.PSANet**

- Related Paper: [click](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/psanet).

**18.APCNet**

- Related Paper: [click](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/apcnet).

**19.FastFCN**

- Related Paper: [click](https://arxiv.org/pdf/1903.11816.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/fastfcn).

**20.UPerNet**

- Related Paper: [click](https://arxiv.org/pdf/1807.10221.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/upernet).

**21.PointRend**

- Related Paper: [click](https://arxiv.org/pdf/1912.08193.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/pointrend).

**22.Deeplabv3**

- Related Paper: [click](https://arxiv.org/pdf/1706.05587.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/deeplabv3).

**23.Segformer**

- Related Paper: [click](https://arxiv.org/pdf/2105.15203.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/segformer).

**24.MaskFormer**

- Related Paper: [click](https://arxiv.org/pdf/2107.06278.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/maskformer).

**25.SemanticFPN**

- Related Paper: [click](https://arxiv.org/pdf/1901.02446.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/semanticfpn).

**26.NonLocalNet**

- Related Paper: [click](https://arxiv.org/pdf/1711.07971.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/nonlocalnet).

**27.Deeplabv3Plus**

- Related Paper: [click](https://arxiv.org/pdf/1802.02611.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/deeplabv3plus).

**28.MemoryNet-MCIBI**

- Related Paper: [click](https://arxiv.org/pdf/2108.11819.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/memorynet).

**29.MemoryNetV2-MCIBI++**

- Related Paper: [click](https://arxiv.org/pdf/2209.04471.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/memorynetv2).

**30.Mixed Precision (FP16) Training**

- Related Paper: [click](https://arxiv.org/pdf/1710.03740.pdf),
- Reported Performance: [click](https://github.com/SegmentationBLWX/sssegmentation/tree/main/docs/performances/fp16).