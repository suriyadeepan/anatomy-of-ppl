# Inference

------

- [x] Enumeration = Search
  - [x] Breadth-first
  - [x] Depth-first
  - [x] Smart Prioritization

- [x] Sampling

  - [x] Particle Filters
  - [x] MCMC

- [ ] Variational Inference

  - [ ] ELBO
  - [ ] Monte-Carlo ELBO

# Enumeration

------

Enumeration is a search strategy for inferring discrete variables i.e. variables that take a value from a finite set of values. Consider a probabilistic program, in which the randomness comes from one or more discrete random variables. Every execution of said program takes different paths. Each combination of these values taken by variables leads to a different execution. Enumeration explores the space of executions of such a probabilistic program. By enumerating all possible executions, we can arrive at a marginal distribution over unknowns. 

The following program is a limited variation of geometric distribution. We sequentially sample from a Bernoulli distribution. 

```python
def geometric_finite():
  a = sample(Bernoulli(p=0.5))
  b = sample(Bernoulli(p=0.5))
  c = sample(Bernoulli(p=0.5))
  return a + b + c
```

The combinations of samples `a`, `b` and `c` are tabulated below.

| a    | b    | c    | a + b + c |
| ---- | ---- | ---- | --------- |
| 0    | 0    | 0    | 0         |
| 0    | 0    | 1    | 1         |
| 0    | 1    | 0    | 1         |
| 0    | 1    | 1    | 2         |
| 1    | 0    | 0    | 1         |
| 1    | 0    | 1    | 2         |
| 1    | 1    | 0    | 2         |
| 1    | 1    | 1    | 3         |

Based on the table, the marginal distribution of `geometric_finite` is given in figure below:

| values | probability |
| ------ | ----------- |
| 0      | 0.125       |
| 1      | 0.375       |
| 2      | 0.375       |
| 3      | 0.125       |

- This is enumeration done manually

> Insert marginal distribution figure.

*Automatic Enumeration* automates this process by exploring different combinations of values returned by `sample` statements while keeping track of unexplored executions. Consider the procedural execution of `geometric_finite`. When the first `sample` statement is encountered, there are two possibilities : `a=0` and `a=1`. Enumeration procedure must keep track of both these values. When the second `sample` statement is encountered, there are 4 possibilities : `00` (`a=0; b=0` ), `01`, `10` and `11`. We repeat the same procedure when we encounter the third `sample` statement. This is basically breadth-first search. Alternatively, we could consider `000` (`a=0; b=0; c=0`), `001`, … `111`, which would be depth-first search.

> Insert depth-first vs breadth-first figure.

In this case, there is no computational difference between the two, since we enumerate over the whole space of executions. This naive enumeration technique will fail when the space of executions becomes large. In case of very large search spaces, where exploration of the whole space is computationally taxing, we need to consider the executions paths (*representative samples*) that would lead to a decent approximation of marginal distribution over unknown(s). We need to score each execution path, explore the ones with higher scores, ignore the low-scoring paths (*exploitation*) and periodically consider random paths (*exploration*). This variant on enumeration with *smart prioritization* is basically a *heuristic search* over the space of executions.

## Sampling

------

*Notes on MCMC*

- Break down probabilities into simple parts

- Simulate samples from parts

- Use samples to represent outcomes from simple probability distributions

- Combine outcomes from simple parts into complex outcomes

- Use counts of simulated outcomes as an estimate of our complex probability distribution

  

- Break down posterior into simple pieces using Bayes Theorem
  
  - Posterior = likelihood x Prior / P(data)
- Simulate samples from parts
  - P(data | model) : Likelihood
  - P(model) : Prior
- Combine outcomes from likelihood and prior into complex outcomes
  
  - We use MCMC for this
- MCMC : Chain of steps through parameter space (space of all the possible models)
  - Step from one model to next is random
  - You spend more time with models that are plausible
    - Metropolis-Hastings Algorithm

> "Monte Carlo is an extremely bad method; it should be used only when all alternative methods are worse."
>
> ~Alan Sokal, 1996

- Enumeration is infeasible for models with large search spaces. 

- We explore a representative subset of execution paths

- Random Sampling : We sample from paths in proportion to their posterior probability

  - Representative picture of marginal distribution

- Importance Sampling

  - Calculate importance weights : $w = f/g$ 
  - where $f$ is the target distribution and $g$ is the proposal distribution
  - Resampling : Draw $N$ samples from $g$ with new weights $w$

  

![](https://machinelearning1.files.wordpress.com/2017/10/is1_1.png?w=316&h=237) ![](https://machinelearning1.files.wordpress.com/2017/10/is1_2.png?w=316&h=237)

- Particle Filtering (Sequential Monte Carlo)
  - Target Distribution is approximated by cloud of random samples (particles)
  - Evolving according to 
    - Importance Sampling
    - Resampling

> How can we improve upon likelihood weighting? Let’s apply the idea from the lecture on [Early, incremental evidence](http://dippl.org/chapters/04-factorseq.html): instead of waiting until the end to resample, we could resample earlier. In particular, we can resample at each factor.
>
> This requires a slight change in our approach. Previously, we ran each sample until the end before we started the next one. Now, we want to run each sample until we hit the first factor statement; resample; run each sample up to the next factor statement; resample; and so on.
>
> To enable this, we store the continuation for each sample so that we can resume computation at the correct point. We are also going to refer to (potentially incomplete) samples as “particles”.

- MCMC

> A popular way to estimate a difficult distribution is to sample from it by constructing a random walk that will visit each state in proportion to its probability – this is called Markov chain Monte Carlo.



Monte Carlo methods estimate an unknown distribution with intractable analytical form by repeatedly sampling from the it. The empirical average over the samples gives a good approximation of the unknown distribution. Markov Chain Monte Carlo (MCMC) is regarded as the golden standard approach for Probabilistic Inference. MCMC builds a markov chain of samples, each sample dependent on the previous sample (or state), to estimate the unknown distribution. Different flavours of MCMC exist. Their differences lie in how the next sample is chosen and varying levels of access to the unknown distribution. For examples, Gibbs Sampling is the most reliable technique for estimating a joint distribution when we are allowed to sample from the conditionals.

*Importance Sampling* is a simple MCMC technique where we approximate the target distribution `p` by repeatedly updating a proposal distribution `g` by re-weighting samples under $p$. The algorithm is presented below:

1. *Sampling* : Draw `S` samples from $g$
2. Calculate probability of each sample under $g$
3. *Evaluation* : Calculate probability of each sample under $p$
4. *Weighting* : Calculate importance weights $w = p / g$
5. *Resampling* : Draw $S$ samples from $g$ with new weights $w$

- Oversampling : weighted weakly
- Under-sampling : weighted strongly

In Importance Sampling, gradually most particles (samples) become useless as they do not match the observations. The importance weights tend towards zero. *Sequential Monte Carlo* or *Particle Filtering* fixes this problem by resampling particles at the end according to new weights. *Low-weighted particles are dropped and high-weighted particles are duplicated*.

*Metropolis-Hasting* sampling is another popular MCMC technique which performs a random walk over the series of samples. We choose an arbitrary probability distribution $g(\overline{x}|x)$ that suggests a candidate for the next sample value $\overline{x}$ given the previous sample value $x$. Typically $g$ is set as a gaussian distribution centered on $x$ so that the points closer to $x$ are more likely to be visited next. This makes the sample generation process, a random walk.

1. *Generate* : Sample from $g(\overline{x} | x)$ to get a candidate $\overline{x}$
2. *Acceptance Ratio* : $\alpha = P(\overline{x}) / P(x)$
3. *Uniform Sample* : $u \sim U(0, 1)$
4. *Accept* : if $u \le \alpha$, $x_{t+1} = \overline{x}$
5. *Reject* : if $u > \alpha$, $x_{t+1} = x_t$

# Variational Inference

------

Variational Inference (VI) is a relatively recent trend in Probabilistic Inference. VI turns inference into an optimization problem. A family of distributions $q$ (variational distribution) with useful properties is chosen and optimized to be closer to the true posterior over unknown. VI trades off accuracy for computation time. Typically we choose Mean-field Exponential family of distributions for $q$.

Any model can be expressed as a joint distribution over observed and latent variables given by $p_{\theta}(x, z) = p(x | z) p(z)$. The joint distribution can be broken down into simpler distributions $\{ p_i \}$. We can sample from these simpler distributions and evaluate their scores under $p_i$. 

The objectives are to find parameters of the model $\theta_{max}$ that fit the data and estimate the posterior over latent variables $z$. By maximizing the log evidence we arrive at the best model parameters.
$$
\theta{max} = argmax_{\theta}\  log\  p_{\theta}(x)\\
log\ p_{\theta}(x) = log \int p_{\theta}(x, z) dz
$$
Posterior over latent variables is given by,
$$
p_{\theta_{max}} (z|x) = \frac{p_{\theta_{max}} (x, z)}{\int p_{\theta_{max}}(x,z)\ dz}
$$
Evidence expressed as the integral over the joint distribution that marginalizes latent variables  $z$ is usually intractable. 

- [ ] KL-Divergence
- [ ] Derivation of ELBO from KL-Divergence

The ELBO objective, a function of both $\theta$ and $\phi$, given by
$$
ELBO = \mathbb{E} \big{[}log\ p_{\theta} (x, z) - log\ q_{\phi} (z) \big{]}
$$
