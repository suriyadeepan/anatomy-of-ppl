import torch
from handlers import trace, block, replay


class Adam(object):
  # dynamically generate optimizers for dynamically generated parameters
  # one optimizer per parameter

  def __init__(self, optim_args):
    self.optim_args = optim_args  # optimizer arguments
    # dictionary of optimizers
    self.optimizers = {}

  def __call__(self, params):
    for param in params:
      if param in self.optimizers:  # have we seen this param before?
        optim = self.optimizers[param]  # select optimizer
      else:  # ELSE create new optimizer for each param
        # with optim arguments like learning rate
        optim = torch.optim.Adam([param], **self.optim_args)
        # add to our dictionary of optimizers
        self.optimizers[param] = optim

      # now that we have selected/created an optimizer
      #  run a step
      optim.step()


class SVI(object):
  # Interface for Stochastic Variational Inference
  # [ model, guide, optimizer, loss_fn ]
  #
  # [1] Run Model and Guide
  # [2] Construct loss function
  # [3] Take a gradient step
  #

  def __init__(self, model, guide, optimizer, loss_fn):
    self.model = model
    self.guide = guide
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def step(self, *args, **kwargs):
    # run trace to capture parameter values
    with trace() as param_capture:
      # block `sample` sites; capture only parameters
      with block(hide_fn=lambda msg : msg['type'] == 'sample'):
        # calculate loss
        loss = self.loss_fn(self.model, self.guide, *args, **kwargs)

    # calculate gradients
    loss.backward()
    # grab parameters from trace
    params = [ site['value'].unconstrained()
        for site in param_capture.values() ]
    # run optimizer; update params
    self.optimizer(params)
    # clear gradients
    for p in params:  # set as zeros
      p.grad = p.new_zeros(p.shape)

    # return loss value
    return loss


def elbo(model, guide, *args, **kwargs):
  # calculate Evidence Lower Bound
  #
  # run guide with args
  # record `sample` and `param` calls
  guide_trace = trace(guide).get_trace(*args, **kwargs)
  # trace model execution
  # replay : reuse sampled values from guide trace
  model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
  # elbo
  elbo = 0.
  #
  # iterate through sample sites in model
  # add log_p(z) term to ELBO
  for site in model_trace.values():
    if site['type'] == 'sample':
      # log p(z)
      elbo = elbo + site['fn'].log_prob(site['value']).sum()
  #
  # iterate through sample sites in guide
  # add log_q(z) term to ELBO
  for site in guide_trace.values():
    if site['type'] == 'sample':
      elbo = elbo - site['fn'].log_prob(site['value']).sum()

  return -elbo  # because we minimize
