# pytorcher
Examples using pytorch
run.py exact copy of of https://pytorch.org/tutorials/intermediate/dist_tuto.html

run_distributed.py spawns one process for each (virtual) core of the node, increments dummy counter and returns empty value to process with global rank 0. This simulates a large process being performed on a large number of nodes