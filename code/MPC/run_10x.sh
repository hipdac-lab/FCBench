#!/bin/bash
RESDIR=/home/cc/experiments
for i in {0..9}
do
	echo "#" $i "================"	>> $RESDIR/mpc_comp.txt
	echo "#" $i "================"	>> $RESDIR/mpc_decomp.txt
	bash test_mpc.sh
done