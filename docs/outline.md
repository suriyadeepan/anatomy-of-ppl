# Outline

<center>Anatomy of Probabilistic Programming Languages</center>


- [ ] Part I

  - Introduction to Probabilistic Programming
  - Utility
  - Applications
  - Model
    - Random Number Generator : `pyro.distributions`
    - Deterministic Computation

- [ ] Part II : Inference

  - Enumeration = Search
    - Breadth-first
    - Depth-first
    - Smart Prioritization
  - Sampling
    - Particle Filters
    - MCMC
  - Variational Inference
    - ELBO
    - Monte-Carlo ELBO

- [ ] Part III : Implementation

  - Continuation Passing Style (CPS)
    - Continuations
    - Co-routines
    - CPS style Factorial Program
    - CPS Model Definition : 3 Bernoulli sample statements (a + b + c)
    - CPS Transform

- [ ] Part IV : Effects and Effect Handlers

  - Exception Handling analogy

    > Each effectful statement wraps itself inside a try-catch block.

  - Effects

    - `sample`
    - `param`

  - Handlers

    - `trace`
    - `replay`
    - `block`

  - Global stack of effect handlers

    - Message Passing up and down the stack

  - Parameter Store

    - Constraints : `torch.distributions.constraints`

- [ ] Part V : Problem Solving

  - Discrete distribution example
    - Bernoulli
    - Enumeration
  - Poisson Regression
  - GDP Regression
  - Logistic Regression on IRIS

- [ ] Part VI : Model Criticism

  - Posterior Predictive Checks
  - Figures

- [ ] Part VII : Latex work

  - Get format from scipy github repository
  - Setup texmaker environment with style files
  - Make it compile
  - Fill in section headers and compile

- [ ] Part VIII : Distributions Library

  - What happens in `torch.distributions` library?
  - Can we use `autograd` to implement a minimal `distributions` library?
  - What does pyro's `distributions` wrapper do?

  

## References

---

- [Author Instructions](<https://github.com/scipy-conference/scipy_proceedings#general-information-and-guidelines-for-authors>)

