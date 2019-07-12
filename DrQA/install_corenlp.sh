#!/bin/bash
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



# By default download to the data directory I guess
read -p "Specify download path or enter to use default (data/corenlp): " path
DOWNLOAD_PATH="${path:-data/corenlp}"
echo "Will download to: $DOWNLOAD_PATH"

# Put jars in DOWNLOAD_PATH
# mv "tmp/stanford-corenlp-full-2017-06-09/"*".jar" "$DOWNLOAD_PATH/"

# Append to bashrc, instructions
while read -p "Add to ~/.bashrc CLASSPATH (recommended)? [yes/no]: " choice; do
    case "$choice" in
        yes )
            echo "export CLASSPATH=\$CLASSPATH:$DOWNLOAD_PATH/*" >> ~/.bashrc;
            break ;;
        no )
            break ;;
        * ) echo "Please answer yes or no." ;;
    esac
done

printf "\n*** NOW RUN: ***\n\nexport CLASSPATH=\$CLASSPATH:$DOWNLOAD_PATH/*\n\n****************\n"
