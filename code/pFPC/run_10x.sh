#!/bin/bash
RESDIR=/home/cc/experiments
for i in {4..9}
do
	echo "#" $i "================"	>> $RESDIR/pfpc_comp.txt
	echo "#" $i "================"	>> $RESDIR/pfpc_decomp.txt
	bash test_pfpc.sh
done