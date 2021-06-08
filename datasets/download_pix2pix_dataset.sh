#!/usr/bin/env bash
FILE=$1

if [[ $FILE != "cityscapes" && $FILE != "maps" ]]; then
  echo "Available datasets are cityscapes, maps"
  exit 1
fi

if [[ $FILE == "cityscapes" ]]; then
  echo "Due to license issue, we cannot provide the Cityscapes dataset from our repository. Please download the Cityscapes dataset from https://cityscapes-dataset.com, and use the script ./datasets/prepare_cityscapes_dataset.py."
  echo "You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. For further instruction, please read ./datasets/prepare_cityscapes_dataset.py"
  exit 1
fi

echo "Specified [$FILE]"

prefix=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

URL=$prefix$FILE.tar.gz
TAR_FILE=./database/$FILE.tar.gz
TARGET_DIR=./database/$FILE/

mkdir -p ./database
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./database/
rm $TAR_FILE

cd "./database/$FILE" || exit

if [ -e "test" ] && [ ! -e "val" ]; then
  ln -s "test" "val"
fi
