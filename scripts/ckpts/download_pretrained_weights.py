'''PRETRAINED_MODEL_WEIGHTS'''
PRETRAINED_MODEL_WEIGHTS = {
    'vit': {
        'jx_vit_large_p16_384': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/jx_vit_large_p16_384-b3be5167.pth',
    },
    'mae': {
        'mae_pretrain_vit_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mae_pretrain_vit_base.pth',
    },
    'mit': {
        'mit-b0': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b0.pth',
        'mit-b1': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b1.pth',
        'mit-b2': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b2.pth',
        'mit-b3': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b3.pth',
        'mit-b4': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b4.pth',
        'mit-b5': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mit_b5.pth',
    },
    'beit': {
        'beit_base_patch16_224_pt22k_ft22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/beit_base_patch16_224_pt22k_ft22k.pth',
        'beit_large_patch16_224_pt22k_ft22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/beit_large_patch16_224_pt22k_ft22k.pth',
    },
    'swin': {
        'swin_tiny_patch4_window7_224': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_tiny_patch4_window7_224.pth',
        'swin_small_patch4_window7_224': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_small_patch4_window7_224.pth',
        'swin_base_patch4_window12_384': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_base_patch4_window12_384.pth',
        'swin_base_patch4_window7_224': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_base_patch4_window7_224.pth',
        'swin_base_patch4_window12_384_22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_base_patch4_window12_384_22k.pth',
        'swin_base_patch4_window7_224_22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_base_patch4_window7_224_22k.pth',
        'swin_large_patch4_window12_384_22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/swin_large_patch4_window12_384_22k.pth',
    },
    'hrnet': {
        'hrnetv2_w18_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/hrnetv2_w18_small-b5a04e21.pth',
        'hrnetv2_w18': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/hrnetv2_w18-00eb2006.pth',
        'hrnetv2_w32': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/hrnetv2_w32-dc9eeb4f.pth',
        'hrnetv2_w40': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/hrnetv2_w40-ed0b031c.pth',
        'hrnetv2_w48': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/hrnetv2_w48-d2186c55.pth',
    },
    'twins': {
        'pcpvt_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/pcpvt_small.pth',
        'pcpvt_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/pcpvt_base.pth',
        'pcpvt_large': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/pcpvt_large.pth',
        'svt_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/alt_gvt_small.pth',
        'svt_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/alt_gvt_base.pth',
        'svt_large': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/alt_gvt_large.pth',
    },
    'resnet': {
        'resnet18': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet18-5c106cde.pth',
        'resnet34': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet34-333f7ec4.pth',
        'resnet50': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet50-19c8e357.pth',
        'resnet101': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet152-b121ed2d.pth',
        'resnet18conv3x3stem': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet18_v1c-b5776b93.pth',
        'resnet50conv3x3stem': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet50_v1c-2cccc1ad.pth',
        'resnet101conv3x3stem': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnet101_v1c-e67eebb6.pth',
    },
    'resnest': {
        'resnest50': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnest50_d2-7497a55b.pth',
        'resnest101': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnest101_d2-f3b931b2.pth',
        'resnest200': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/resnest200_d2-ca88e41f.pth',
    },
    'convnext': {
        'convnext_tiny': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
        'convnext_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',
        'convnext_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
        'convnext_base_21k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
        'convnext_large_21k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth',
        'convnext_xlarge_21k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth',
        'convnextv2_atto_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_atto_1k_224_fcmae.pt',
        'convnextv2_femto_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_femto_1k_224_fcmae.pt',
        'convnextv2_pico_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_pico_1k_224_fcmae.pt',
        'convnextv2_nano_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_nano_1k_224_fcmae.pt',
        'convnextv2_tiny_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_tiny_1k_224_fcmae.pt',
        'convnextv2_base_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_base_1k_224_fcmae.pt',
        'convnextv2_large_1k_224_fcmae': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_large_1k_224_fcmae.pt',
        'convnextv2_huge_1k_224_fcmae': [
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_fcmae.zip',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_fcmae.z01',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_fcmae.z02',
        ],
        'convnextv2_atto_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_atto_1k_224_ema.pt',
        'convnextv2_femto_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_femto_1k_224_ema.pt',
        'convnextv2_pico_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_pico_1k_224_ema.pt',
        'convnextv2_nano_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_nano_1k_224_ema.pt',
        'convnextv2_tiny_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_tiny_1k_224_ema.pt',
        'convnextv2_base_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_base_1k_224_ema.pt',
        'convnextv2_large_1k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_large_1k_224_ema.pt',
        'convnextv2_huge_1k_224_ema': [
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_ema.zip.001',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_ema.zip.002',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_1k_224_ema.zip.003',
        ],
        'convnextv2_nano_22k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_nano_22k_224_ema.pt',
        'convnextv2_nano_22k_384_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_nano_22k_384_ema.pt',
        'convnextv2_tiny_22k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_tiny_22k_224_ema.pt',
        'convnextv2_tiny_22k_384_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_tiny_22k_384_ema.pt',
        'convnextv2_base_22k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_base_22k_224_ema.pt',
        'convnextv2_base_22k_384_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_base_22k_384_ema.pt',
        'convnextv2_large_22k_224_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_large_22k_224_ema.pt',
        'convnextv2_large_22k_384_ema': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_large_22k_384_ema.pt',
        'convnextv2_huge_22k_384_ema': [
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_384_ema.zip.001',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_384_ema.zip.002',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_384_ema.zip.003',
        ],
        'convnextv2_huge_22k_512_ema': [
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_512_ema.zip.001',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_512_ema.zip.002',
            'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/convnextv2_huge_22k_512_ema.zip.003',
        ],
    },
    'mobilenet': {
        'mobilenetv2': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
        'mobilenetv3_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilenet_v3_small-47085aa1.pth',
        'mobilenetv3_large': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilenet_v3_large-bc2c3fd3.pth',
    },
    'mobilevit': {
        'mobilevit-small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth',
        'mobilevit-xsmall': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth',
        'mobilevit-xxsmall': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth',
        'mobilevitv2_050': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_050-49951ee2.pth',
        'mobilevitv2_075': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_075-b5556ef6.pth',
        'mobilevitv2_100': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_100-e464ef3b.pth',
        'mobilevitv2_125': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_125-0ae35027.pth',
        'mobilevitv2_150': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_150-737c5019.pth',
        'mobilevitv2_175': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_175-16462ee2.pth',
        'mobilevitv2_200': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_200-b3422f67.pth',
        'mobilevitv2_150_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_150_in22ft1k-0b555d7b.pth',
        'mobilevitv2_175_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_175_in22ft1k-4117fa1f.pth',
        'mobilevitv2_200_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_200_in22ft1k-1d7c8927.pth',
        'mobilevitv2_150_384_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_150_384_in22ft1k-9e142854.pth',
        'mobilevitv2_175_384_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_175_384_in22ft1k-059cbe56.pth',
        'mobilevitv2_200_384_in22ft1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/mobilevitv2_200_384_in22ft1k-32c87503.pth',
    },
    'mobilesamtinyvit': {
        'tiny_vit_5m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_5m_1k.pth',
        'tiny_vit_11m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_11m_1k.pth',
        'tiny_vit_21m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_21m_1k.pth',
        'tiny_vit_5m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_5m_22k_distill.pth',
        'tiny_vit_11m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_11m_22k_distill.pth',
        'tiny_vit_21m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_21m_22k_distill.pth',
        'tiny_vit_5m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_5m_22kto1k_distill.pth',
        'tiny_vit_11m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_11m_22kto1k_distill.pth',
        'tiny_vit_21m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_21m_22kto1k_distill.pth',
        'tiny_vit_21m_22kto1k_384_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_21m_22kto1k_384_distill.pth',
        'tiny_vit_21m_22kto1k_512_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pretrained/tiny_vit_21m_22kto1k_512_distill.pth',
    },
}


'''download'''
def download(backbone_type, structure_type, disable_logging=True, logger_handle=None):
    import os
    import subprocess
    link = PRETRAINED_MODEL_WEIGHTS[backbone_type][structure_type]
    if (logger_handle is not None) or (not disable_logging):
        if logger_handle is not None:
            logger_handle.info(f'Download {link} by giving backbone_type {backbone_type} and structure_type {structure_type}')
        else:
            print(f'Download {link} by giving backbone_type {backbone_type} and structure_type {structure_type}')
    if isinstance(link, str):
        subprocess.run(f'wget {link}', shell=True, check=True)
    else:
        assert isinstance(link, list)
        filenames = []
        for l in link:
            filenames.append(l.split('/')[-1])
            subprocess.run(f'wget {l}', shell=True, check=True)
        subprocess.run(f'7z x {filenames[0]}', shell=True, check=True)
        for filename in filenames: os.remove(filename)


'''DEBUG'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scripts for downloading pretrained backbone model weights.')
    parser.add_argument('--backbone', dest='backbone', help='backbone type, e.g., resnet and swin', default='resnet', type=str)
    parser.add_argument('--structure', dest='structure', help='structure type, e.g., resnet101conv3x3stem and swin_large_patch4_window12_384_22k', default='resnet101conv3x3stem', type=str)
    args = parser.parse_args()
    download(args.backbone, args.structure, disable_logging=False)