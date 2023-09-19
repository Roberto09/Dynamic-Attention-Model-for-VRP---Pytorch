# Dynamic-Attention-Model-for-VRP---Pytorch

This is an implementation for the paper "A Deep Reinforcement Learning Algorithm Using Dynamic Attention Model for Vehicle Routing Problems" (https://arxiv.org/abs/2002.03282) done in Pytorch and based on a tensorflow implementation https://github.com/d-eremeev/ADM-VRP.

Pytorch is well known to be faster than Tensorflow in many cases if used properly. Here we noticed increases of +400% speed over Tensorflow implementations in our tests without the memory-efficient gradient trick (and a bit less with it).

To run take a look at **./test_20n_1024bs.ipynb**. This simple notebook was used to test 40 epochs of CVRP_20 following the description proposed by Nazari et. al (as done in the paper) with batch sizes of 1024. Results for this notebook (except for the valsets and checkpts are included for demonstrative purposes; ex. take a look at ./backup_results_VRP_20_2021-08-31.csv.

About the memory-efficient gradient trick:

I noticed that we could rearrange the formula of the gradients being used such that when translating it to code we sacrifice runtime in order to stop storing huge computation graphs in cuda memory. This trick significantly reduces the memory being used; if you want to see exactly what it is please take a look into the commit itself (https://github.com/Roberto09/Dynamic-Attention-Model-for-VRP---Pytorch/commit/fc1f9a8b6650fcd3cae23fb18db826147a29c3a1). The explanation of how/why this works can be found in the "Gradient Computation Improvement for AM-D" section of a paper we wrote (https://arxiv.org/abs/2211.13922).
