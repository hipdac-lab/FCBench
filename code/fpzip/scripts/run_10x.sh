#!/bin/bash
RESDIR=/home/cc/experiments
for i in {0..9}
do
	echo "#" $i "================"	>> $RESDIR/fpzip_comp.txt
	echo "#" $i "================"	>> $RESDIR/fpzip_decomp.txt
	bash test_fpzip.sh
done