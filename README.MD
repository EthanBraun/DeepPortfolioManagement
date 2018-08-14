# Deep Portfolio Management

### Background
This repo contains an implementation of the paper [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059.pdf). The paper describes a cryptocurrency portfolio managing agent consisting of a fully convolutional neural network trained through reinforcement learning. The policy network, called the Ensemble of Identical Independent Evaluators (EIIE), takes a 3D tensor of historical market-relative price data as input, as well as the vector of portfolio weights from the previous period. The output of the EIIE is the portfolio weight vector for the subsequent period. 

![Ensemble of Identical Independent Evaluators](https://i.imgur.com/vbebH1c.png)

The implementation is written in Python, using Keras and Tensorflow to construct the policy network and calculate its gradient.

### Framework
##### Training
The EIIE is trained at the end of each trading period on stochastic mini-batches of historical data.
Mini-batch intervals are sampled from a geometric distribution so recent data is selected more often than older data. The agent maintains a memory of the portfolio weight vector at each trading period, known as the Portfolio Vector Memory (PVM). The PVM is overwritten both when the agent is redistributing the portfolio and during training. For each mini-batch, the agent ascends the reward gradient of the interval by an amount determined by the learning rate $\lambda$. The reward gradient with respect to the EIIE weights $\theta _t$ of a trading period $t$ is the logarithmic rate of return for the period given by the formula, 
$\nabla r_{\theta t}=ln(\mu_t y_t\cdot w_ {t-1})$
where $w_{t-1}$ is the portfolio weight vector at the start of period $t$, $y_t$ is the market-relative price vector for period $t$, and $\mu _t$ is the factor by which the portfolio decreases in value due to trading fees in redistribution for the period.
