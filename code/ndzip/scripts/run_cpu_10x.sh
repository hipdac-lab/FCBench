#!/bin/bash
RESDIR=/home/cc/experiments
for i in {1..9}
do
	echo "#" $i "================"	>> $RESDIR/ndzip_comp_cpu.txt
	echo "#" $i "================"	>> $RESDIR/ndzip_decomp_cpu.txt
	bash batch_ndzip_cpu.sh
done