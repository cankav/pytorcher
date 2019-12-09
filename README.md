# pytorcher
Examples using pytorch

run.py exact copy of of https://pytorch.org/tutorials/intermediate/dist_tuto.html

run_distributed.py spawns one process for each (virtual) core of the node. Each process increments dummy counter (i.e performs a large computation) and returns calculated value to process with global rank 0. This outlines a distributed architecture for a large process being performed on a large number of nodes using different parts of a large dataset.