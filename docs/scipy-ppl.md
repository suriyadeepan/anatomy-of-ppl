# Concepts in PPL

------



- What is probabilistic programming?
- What is this obsession with recursion?

Probabilistic Programming allows us to describe a model as a program, condition the execution of the program on some data and provides a distribution over the outputs of the model. The process of inferring a distribution over the outputs, is known as inference or marginalization. PPLs typically abstract away the inference procedure from the programmer.

**What are higher-order functions?**

A higher-order function or a functional is a function that does one of the following:

- Takes one or more function as argument
- Return a function as its result

**What makes a language universal?**

A PPL is called universal if it can be used to represent any computable probability distribution. The language must support two features in order to be universal.

- Deterministic Computation
  - Loops, Recursion, Higher-order functions
- Random Control Flow
  - Random number generator

Therefore, a PPL can be built on top of any (host) language that supports deterministic computation and random number generation. WebPPL is built on top of a subset of javascript by Noah Goodman and peers. In this series of experiments, we will attempt to build a minimal PPL using python as the host language.

**What is this stochastic function you speak of? How do you build one with "randomness"?**

> Consider a sequence of trials, where each trial has only two possible outcomes (designated failure and success). The probability of success is assumed to be the same for each trial. In such a sequence of trials, the geometric distribution is useful to model the number of failures before the first success. 

```python
def geometric(n=0):
  x = sample(Bernoulli(p=0.5))
  if x == 1:
    return n
  else:
    return geometric(n+1)
```

- `geometric` is a stochastic function that combines randomness, conditionals and recursion. 

**Garden of Forking Paths**

- *Insert garden of forking paths picture*
- Every time you sample, you are choosing a path of execution. Multiple paths collapse into one.
- `Bernoulli` is a primitive stochastic function from `distributions` module, which is built based on random number generator.

**What makes a distribution?**

- Distribution represents a parameterized probability distribution 
- A distribution must support three operations
  - `sample` : we should be able to sample from the distribution
  - `score` : returns the log-probability of a value possibly sampled from the distribution
  - `support` : list of all the values supported by the distribution. This makes sense for a discrete distribution like Bernoulli - 0 and 1 are the possible outcomes. How do you return the support of a `gaussian` distribution?

- Insert figure demonstrating `sample`, `support` and `score`

**How do we infer the distribution over the outputs of `geometric` model?**

Imagine we execute the function `geometric`. We sample from `Bernoulli` over and over, until we arrive at a `1`. At which point, the number of trials so far (to arrive at a `1`), is returned and the execution terminates. This basically gives us the outcome of one particular path of execution. For example, the output `3` implies, `sample` statement yielded `0001`, ie. three failures before a success. What we want is a probability distribution over the outcome of `geometric` function. Something like this:

- *Insert probability distribution figure*

For inferring such a distributions, we should follow all the execution paths. And score the output based on which ones are more likely. If we follow all the execution paths of `geometric` distribution, we'll end up in an infinite loop. Let's consider a simpler example.

```python
def geometric_finite():
  a = sample(Bernoulli(p=0.5))
  b = sample(Bernoulli(p=0.5))
  c = sample(Bernoulli(p=0.5))
  return a + b + c
```

In this case, we've limited the experiment to just 3 trials. The possible outcomes are 0, 1, 2, 3. We need to score each outcome. How we do that is the business of the inference procedure.

**Enumeration**

- Continuation 




