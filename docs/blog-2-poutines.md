# Effect Handlers

------

What are effects?

Analyse, name, restrict what a program can do.

- Effects and Handlers
- Generalization of exceptions - can resume
- Lower-level code signals condition
- Higher-level code handles conditions and chooses restart
- Control returns to lower-level code to execute restart
- Idea : Save, don't discard, reminder of computation to do!
- Continuations
  - `1 + sqrt(3 + 4)`
  - `k = lambda v : 1 + sqrt(v)`
- Non-determinism

## Poutine

------

- Inference
  - Manipulating or transforming probabilistic programs
  - Compute unnormalized log probability of values of latent and observed variables

```python
def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))
```

- We need access to inputs and outputs of `pyro.sample` site, to compute *log-joint*

- `scale` doesn't expose the intermediate distribution objects for us to calculate log-joint

- *Poutine* is a library of effect handlers or composable building blocks for examining and modifying the behaviour of pyro programs

- Non-standard interpretations or side-effects to the behaviour of particular statements (`pyro.sample`, `pyro.param`)

- Composability

  - `poutine.condition` sets output values of `pyro.sample` statements
  - `poutine.trace` records the inputs, distributions and outputs of `pyro.sample` statements
  - Combine the two to compute log-joint

  ```python
  def make_log_joint(model):
    def _calc_log_joint(cond_data, *args, **kwargs):
      # condition model on data
      conditioned_model = poutine.condition(model, data=cond_data)
      # get trace
      trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)
      # get sum of log_prob
      return trace.log_prob_sum()
    return _calc_log_joint(scale)
  
  scale_log_joint = make_log_joint(scale)
  print(scale_log_joint({"measurement" : 9.5, "weight" : 8.23}, 8.5))  # -3.02
  ```

- `sample` sites are the only points of contacts for handlers with the model

- `poutine.trace` produces a data structure `Trace`

  - contains a dictionary `{ 'sample_x' : {'fn' : Normal(.), 'output' : 9.5} }`
  - Note that `9.5` matches our data

- Pyro's effect handlers are *Messengers*

  - Stateful Context Manager objects
  - Placed on a global stack
  - Send messages up and down the stack at each effectful operation
  - A *Messenger* is placed at the bottom of the stack when its `__enter__` method is called (`with` statement)