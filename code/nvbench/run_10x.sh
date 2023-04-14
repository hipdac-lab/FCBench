#!/bin/bash
RESDIR=/home/cc/experiments
for i in {1..9}
do
	echo "#" $i "================"	>> $RESDIR/nvlz4.txt
	echo "#" $i "================"	>> $RESDIR/nvbitcomp.txt
	bash batch_nvcomp.sh
done