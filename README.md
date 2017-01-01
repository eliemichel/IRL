Inverse Reinforcement Learning
==============================

This is a project for the lecture [*Graphs in Machine Learning*](http://researchers.lille.inria.fr/~valko/hp/mva-ml-graphs.php), by [Michal Valko](http://researchers.lille.inria.fr/~valko/hp/).

It is developped by Charles Reizine and [Élie Michel](http://www.eleves.ens.fr/home/elmichel/), under the supervision of [Paul Weng](http://weng.fr/index.html).


Description
-----------

Here is the problem description that we advocate in this project:

> In inverse reinforcement learning (IRL), the goal is to learn a reward function that explains the observed demonstrations from a supposedly optimal policy. It is well-known that this problem is ill-posed and under-constrained. In practice, even though the reward function may not be known, some preferential information may be accessible: order over rewards, structural constraints, symmetry… Such information may also be useful in particular to detect that the demonstrated policy may not be optimal. The goal in this project is to study cases where this additional information would make IRL a better (if not a well) posed problem, propose solving algorithms and evaluate them experimentally.


Ongoing work
------------

grid.py: Quick implementation of the first toy problem presented in (1).


Technical Requirements
----------------------

Python, with modules `numpy`, `sklearn`, `matplotlib`.


Resources
---------

1. [A. Y. Ng. & S. Russel. Algorithms for Inverse Reinforcement Learning. ICML 2000](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

2. [P. Abeel & A. Y. Ng. Apprenticeship Learning via Inverse Reinforcement Learning. ICML 2004](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)

3. [R. S. Sutton & A. G. Barto. Reinforcement Learning: An Introduction (draft). 2014-2016](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

4. [A. Y. Ng, D. Harada & S. Russel. Policy invariance under reward transformations: Theory and application to reward shaping. ICML 1999](http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf)

5. [D. Slater. Deep-Q learning Pong with Tensorflow and PyGame. Blog post](http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html)
