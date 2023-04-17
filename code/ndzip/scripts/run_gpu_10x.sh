#!/bin/bash
RESDIR=/home/cc/experiments
for i in {1..9}
do
	echo "#" $i "================"	>> $RESDIR/ndzip_comp_gpu.txt
	echo "#" $i "================"	>> $RESDIR/ndzip_decomp_gpu.txt
	bash batch_ndzip_gpu.sh
done