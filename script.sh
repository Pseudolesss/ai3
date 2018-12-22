#!/bin/bash

clear

w=1
while [ $w -le 5 ]
do
    x=10
    while [ $x -le 90 ]
    do
        i=0
        while [ $i -le 99 ]
        do
            python run.py --bsagentfile beliefstateagent.py --layout observer --w $w --p 0.$x --nghosts 1 --ghostagent leftrandy
        ((i=$i+1))
        done
    ((x=$x+80))
    done
    ((w=$w+2))
done

