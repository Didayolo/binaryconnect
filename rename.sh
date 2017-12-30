#!/bin/sh

CPT=1
for FILENAME in *
do 
	if [ $FILENAME != "rename.sh" ]
	then
		mv $FILENAME $1_$CPT.${FILENAME##*.}
		CPT=$(($CPT + 1))
	fi
done 
