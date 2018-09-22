# rl_transfer

## Introduction:

Reinforcement Learning has shown tremendous success in simulator environments. However, one typical challenge is how to transfer the learned knowledge from the simulator to the real world effectively. Domain transfer in RL consists of representation transfer and policy transfer. In this paper, we focus on the representation transfer for vision based applications, that is, aligning the feature representation from the source domain to target domain in an unsupervised way. Our proposed algorithm uses a linear mapping function to fuse modules that are trained in different domains. We propose two improved adversarial learning methods to enhance the training quality of the mapping function. Finally, we demonstrate the algorithm's effectiveness by using CIFAR-10 and the CARLA autonomous driving environment.

the code is develped based on MUSE from FAIR.



