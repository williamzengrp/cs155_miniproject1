#!/bin/bash
G=./work/gbm
cd $G
Q2=$(cat queue2.pid)
while kill -0 $Q2 2>/dev/null; do sleep 10; done
echo "queue3 start $(date)" >> queue.log
bash plan3.sh
echo "queue3 end $(date)" >> queue.log
