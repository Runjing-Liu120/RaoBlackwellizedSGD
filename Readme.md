# Rao-Blackwellized Stochastic Gradients for Discrete Distributions

This repository contains the implementation of the method and experiments described in 

https://arxiv.org/abs/1810.04777 

By Runjing Liu, Jeffrey Regier, Nilesh Tripuraneni, Michael I. Jordan, Jon McAuliffe

This paper is concerned with optimization objectives of the form 
\begin{align}
E_{q_\eta(z)}[f(z)] = \sum_{k = 1}^K q_\eta(z = k) f(k)
\end{align}
where $z$ is a discrete random variable, so the re-parametrization trick does not apply. 

Many such stochastic gradient estimators have been proposed to run SGD on the above objective, such as RELAX
(Grathwohl et al., 2018), REBAR (Tucker et al., 2017),
NVIL (Mnih & Gregor, 2014), and Gumbel-softmax (Jang et al., 2017). 

In our paper, we describe a technique that can
be applied to reduce the variance of any such
estimator, without changing its bias. 

Our idea is simple. Let $g$ be any unbiased estimate of the gradient 
so 
Let $g(z)$ be any unbiased estimate of the gradient so that
\begin{align}
\nabla_{\eta} E_{q_{\eta}(z)} [f_\eta(z)] = E_{q_{\eta}(z)}[g(z)] = \sum_{k = 1}^K q_\eta(k)g(k)
\end{align}
\pause
Since $K$ might be large, the last sum is difficult to compute.

A simple estimate is to sample $z\sim q_\eta(z)$ and evaluate $g(z)$.
But this might have high variance.

In many applications, $q_\eta(z)$ is concentrated on only a few categories.

**Our idea:** Let us analytically sum categories where $q_\eta(z)$ has high probability,
and sample the remaining terms.

Let $\mathcal{C}_\alpha$ be the categories where $q_\eta(z)$ has probability larger than $\alpha$. Then the exact gradient
can be written as
\begin{align}
 \sum_{k = 1}^K q_\eta(k)g(k) = \sum_{z \in \mathcal{C}_\alpha} q_\eta(z)g(z) + (1 - q(\mathcal{C}_\alpha))E_{q_{\eta}(z)} [f_\eta(z) | z \notin \mathcal{C}_\alpha]
\end{align}
\pause
We compute the first sum analytically, and estimate the conditional expectation by sampling.

\pause
If $q_\eta(z)$ is concentrated around a few categories, then $1 - q(\mathcal{C}_\alpha)$ is small,
and there is almost no variance in the second term. The first term nearly recovers the true gradient.

This intuition is made precise using a **Rao-Blackwellization** argument. See our paper for more details. 

## Some results
This repository reproduces the results shown in the paper. Our implementation of this method can be found in `./rb_utils/`. We also implemented REBAR/RELAX, Gumbel-softmax, and NVIL for comparison. Our experiments can be found in the `./experiments/` folder. Two are highlighted here. 

One experiment examined our performance on a semi-supervised MNIST task (Kingma et al., 2014). Here, the discrete random variable is the digit label. We compare Rao-Blackwellizing the simple REINFORCE estimator against other SOTA methods. See our paper for details of the methods implemented. 

![Comparison of our method (red) against other SOTA methods on the semi-supervised MNIST task, ](./experiments/icml_figures/ss_mnist_elbo_plot.png)

We also trained on a pixel attention task, where we had to locate an MNIST digit randomly placed on a 68 x 68 background. The discrete random variable hence takes on $68^2$ categories. 


![Comparison of our method (red) against other SOTA methods on the moving MNIST task. ](./experiments/icml_figures/moving_mnist_elbo_plot.png)



