#!/bin/bash
RESDIR=/home/cc/experiments
for i in {0..9}
do
	echo "#" $i "================"	>> $RESDIR/gfc_comp.txt
	echo "#" $i "================"	>> $RESDIR/gfc_decomp.txt
	bash test_GFC.sh
done