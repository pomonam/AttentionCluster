# AttentionCluster
This code implements attention clusters with shifting operation. It was developed on top of starters code provided by Google AI. Detailed table of contents and descriptions can be found at the [original repository](https://github.com/google/youtube-8m).

The module was implemented & tested in TensorFlow 1.8.0. Attention Cluster is distributed under Apache-2 License (see the `LICENCE` file). 

# Differences with the original paper
- I used youtube-8m dataset, whereas the literature uses Flash-MNIST.
- Empirically, I found that batch normalization layer at attention mechanism increases the convergence time & GAP.
- In between MoE, I used wide context gating developed from [2]. 

# References
Please note that we are **not** the author of the following references.

[1] https://arxiv.org/abs/1711.09550
[2] https://arxiv.org/abs/1706.06905

# Changes
- **1.00** (05 August 2018)
    - Initial public release
    
