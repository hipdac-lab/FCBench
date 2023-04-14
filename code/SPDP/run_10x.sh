#!/bin/bash
RESDIR=/home/cc/experiments
for i in {2..9}
do
	echo "#" $i "================"	>> $RESDIR/spdp_comp.txt
	echo "#" $i "================"	>> $RESDIR/spdp_decomp.txt
	bash test_spdp.sh
done