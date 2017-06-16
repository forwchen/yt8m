# Code for model fusion

The directory contains scripts for making model ensembles.

An overview of what the scripts do:

| Script | What it does |
| --- | --- |
|fuse_final.sh|Generate our final ensemble.|
|fuse_sxchen.sh|Generate team Shaoxiang Chen's ensemble.|
|select_top.py|For a pridction file, select top k classes for each sample.|
|simple_fusion.py|Generate team Xi Wang's ensemble.|
|valid.py|Generate team Yongyi Tang's ensemble.|
|weighted_fuse.py|Linear weighted fusion code from Shaoxiang Chen.|
|weighted_fuse_count.py|Linear weighted fusion with a counting trick from Shaoxiang Chen.|

For more detailed explanations, see the comments in each file.
