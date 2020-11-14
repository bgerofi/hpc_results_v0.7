#!/bin/bash

DATA_DIR=$1
FILENAME_FILE=$2

NLINES=`wc -l $FILENAME_FILE | cut -d " " -f 1`
LOCAL_NLINES=$(( $NLINES / $OMPI_COMM_WORLD_SIZE ))
BEGIN_LINE=$(( $LOCAL_NLINES * $OMPI_COMM_WORLD_RANK + 1 ))
END_LINE=$(( $LOCAL_NLINES * ($OMPI_COMM_WORLD_RANK + 1) ))

cd $DATA_DIR

sed -n ${BEGIN_LINE},${END_LINE}p $FILENAME_FILE  | xargs -n 1 md5sum
