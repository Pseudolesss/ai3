#!/bin/bash

clear

w=1
while [ $w -le 5 ]
do
    x=25
    while [ $x -le 75 ]
    do
        python run.py --bsagentfile beliefstateagent.py --layout observer --w $w --p 0.$x --nghosts 1 &
        sleep 40 && kill $!
        ((x=$x+25))
    done
    ((w=$w+2))
done

