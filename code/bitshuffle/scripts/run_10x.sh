#!/bin/bash
RESDIR=/home/cc/experiments
for i in {1..9}
do
	echo "#" $i "================"	>> $RESDIR/bitshuffle_all.txt
	bash test_bitshuffle.sh
done