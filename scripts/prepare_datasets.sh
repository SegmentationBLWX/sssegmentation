#!/usr/bin/env bash

help() {
    echo "------------------------------------------------------------------------------------"
    echo "$0 - prepare datasets for training and inference of SSSegmentation."
    echo "------------------------------------------------------------------------------------"
    echo "Usage:"
    echo "    bash $0 <dataset name>"
    echo "Options:"
    echo "    <dataset name>: The dataset name you want to download and prepare."
    echo "                    The keyword should be in ['ade20k', 'lip', 'pascalcontext', 'cocostuff10k',"
    echo "                                              'pascalvoc', 'cityscapes', 'atr', 'chase_db1',"
    echo "                                              'cihp', 'hrf', 'drive', 'stare', 'nighttimedriving',"
    echo "                                              'darkzurich', 'sbushadow', 'supervisely', 'vspw',"
    echo "                                              'mhpv1', 'mhpv2', 'coco',]"
    echo "    <-h> or <--help>: Show this message."
    echo "Examples:"
    echo "    If you want to fetch ADE20k dataset, you can run 'bash $0 ade20k'."
    echo "    If you want to fetch Cityscapes dataset, you can run 'bash $0 cityscapes'."
    echo "------------------------------------------------------------------------------------"
    exit 0
}

DATASET=$1
OPT="$(echo $DATASET | tr '[:upper:]' '[:lower:]')"
if [ "$OPT" == "-h" ] || [ "$OPT" == "--help" ] || [ "$OPT" == "" ]; then
    help
elif [[ "$OPT" == "ade20k" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/ADE20k.tar.gz
        tar zxvf ADE20k.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf ADE20k.tar.gz
elif [[ "$OPT" == "lip" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LIP.tar.gz
        tar zxvf LIP.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf LIP.tar.gz
elif [[ "$OPT" == "pascalcontext" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/PascalContext.tar.gz
        tar zxvf PascalContext.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf PascalContext.tar.gz
elif [[ "$OPT" == "cocostuff10k" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCOStuff10k.tar.gz
        tar zxvf COCOStuff10k.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf COCOStuff10k.tar.gz
elif [[ "$OPT" == "pascalvoc" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.003
        7z x VOCdevkit.zip.001
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf VOCdevkit.zip.001 VOCdevkit.zip.002 VOCdevkit.zip.003
elif [[ "$OPT" == "cityscapes" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.zip
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z01
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z02
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z03
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z04
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z05
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z06
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z07
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z08
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z09
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z10
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z11
        7z x CityScapes.zip
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf CityScapes.zip CityScapes.z01 CityScapes.z02 CityScapes.z03 CityScapes.z04 \
           CityScapes.z04 CityScapes.z05 CityScapes.z06 CityScapes.z07 CityScapes.z08 \
           CityScapes.z09 CityScapes.z10 CityScapes.z11 
elif [[ "$OPT" == "atr" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/ATR.tar.gz
        tar zxvf ATR.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf ATR.tar.gz
elif [[ "$OPT" == "chase_db1" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CHASE_DB1.tar.gz
        tar zxvf CHASE_DB1.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf CHASE_DB1.tar.gz
elif [[ "$OPT" == "cihp" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CIHP.tar.gz
        tar zxvf CIHP.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf CIHP.tar.gz
elif [[ "$OPT" == "hrf" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/HRF.tar.gz
        tar zxvf HRF.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf HRF.tar.gz
elif [[ "$OPT" == "drive" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/DRIVE.tar.gz
        tar zxvf DRIVE.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf DRIVE.tar.gz
elif [[ "$OPT" == "stare" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/STARE.tar.gz
        tar zxvf STARE.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf STARE.tar.gz
elif [[ "$OPT" == "nighttimedriving" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/NighttimeDrivingTest.zip
        unzip NighttimeDrivingTest.zip
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf NighttimeDrivingTest.zip
elif [[ "$OPT" == "darkzurich" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Dark_Zurich_val_anon.zip
        unzip Dark_Zurich_val_anon.zip
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf Dark_Zurich_val_anon.zip
elif [[ "$OPT" == "sbushadow" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/SBUShadow.tar.gz
        tar zxvf SBUShadow.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf SBUShadow.tar.gz
elif [[ "$OPT" == "supervisely" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.003
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.004
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.005
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.006
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.007
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/Supervisely.zip.008
        7z x Supervisely.zip.001
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf Supervisely.zip.001 Supervisely.zip.002 Supervisely.zip.003 Supervisely.zip.004 Supervisely.zip.005 \
           Supervisely.zip.006 Supervisely.zip.007 Supervisely.zip.008
elif [[ "$OPT" == "vspw" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.003
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.004
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.005
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.006
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.007
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.008
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.009
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VSPW_480p.zip.010
        7z x VSPW_480p.zip.001
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf VSPW_480p.zip.001 VSPW_480p.zip.002 VSPW_480p.zip.003 VSPW_480p.zip.004 VSPW_480p.zip.005 \
           VSPW_480p.zip.006 VSPW_480p.zip.007 VSPW_480p.zip.008 VSPW_480p.zip.009 VSPW_480p.zip.010
elif [[ "$OPT" == "coco" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.003
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.004
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.005
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.006
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.007
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.008
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.009
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.010
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.011
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.012
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.013
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.014
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.015
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.016
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.017
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/COCO.zip.018
        7z x COCO.zip.001
        cd COCO
        unzip stuffthingmaps_trainval2017.zip
        cd ..
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf COCO.zip.001 COCO.zip.002 COCO.zip.003 COCO.zip.004 COCO.zip.005 COCO.zip.006 COCO.zip.007 \
           COCO.zip.008 COCO.zip.009 COCO.zip.010 COCO.zip.011 COCO.zip.012 COCO.zip.013 COCO.zip.014 \
           COCO.zip.015 COCO.zip.016 COCO.zip.017 COCO.zip.018
elif [[ "$OPT" == "mhpv1" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LV-MHP-v1.tar.gz
        tar zxvf LV-MHP-v1.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf LV-MHP-v1.tar.gz
elif [[ "$OPT" == "mhpv2" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LV-MHP-v2.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LV-MHP-v2.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LV-MHP-v2.zip.003
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/LV-MHP-v2.zip.004
        7z x LV-MHP-v2.zip.001
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf LV-MHP-v2.zip.001 LV-MHP-v2.zip.002 LV-MHP-v2.zip.003 LV-MHP-v2.zip.004
else
    echo "Preparing dataset ${DATASET} is not supported in this script now."
    exit 0
fi
echo "Download ${DATASET} done."