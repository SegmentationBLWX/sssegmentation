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
    echo "                                              'darkzurich', 'sbushadow', 'supervisely', ]"
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
        zip CityScapes.zip CityScapes.z01 CityScapes.z02 CityScapes.z03 CityScapes.z04 \
            CityScapes.z04 CityScapes.z05 CityScapes.z06 CityScapes.z07 CityScapes.z08 \
            CityScapes.z09 CityScapes.z10 CityScapes.z11 -s=0 --out CityScapes_Merged.zip
        unzip -o CityScapes_Merged.zip
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf CityScapes.zip CityScapes.z01 CityScapes.z02 CityScapes.z03 CityScapes.z04 \
           CityScapes.z04 CityScapes.z05 CityScapes.z06 CityScapes.z07 CityScapes.z08 \
           CityScapes.z09 CityScapes.z10 CityScapes.z11 CityScapes_Merged.zip
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
else
    echo "Preparing dataset ${DATASET} is not supported in this script now."
    exit 0
fi
echo "Download ${DATASET} done."