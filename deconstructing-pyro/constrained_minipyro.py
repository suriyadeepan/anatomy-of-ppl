from collections import OrderedDict
import weakref

import pyro.distributions as pdist
import torch

STACK = []
PARAM_STORE = {}


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
    return loss.item()


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


class Messenger(object):

  def __init__(self, fn=None):
    self.fn = fn

  def __enter__(self):
    STACK.append(self)

  def __exit__(self, *args, **kwargs):
    assert STACK[-1] is self
    STACK.pop()

  def __call__(self, *args, **kwargs):
    with self:
      return self.fn(*args, **kwargs)

  def process_msg(self, msg):
    pass

  def postprocess_msg(self, msg):
    pass


class trace(Messenger):

  def __enter__(self):
    super(trace, self).__enter__()  # execute parent; add to STACK
    self.trace = OrderedDict()  # init trace
    return self.trace  # empty trace

  def postprocess_msg(self, msg):
    assert msg['name'] not in self.trace  # make sure msg has unique name
    self.trace[msg['name']] = msg.copy()  # record msg

  def get_trace(self, *args, **kwargs):
    self(*args, **kwargs)  # execute parent's __call__
    return self.trace  # return trace dict


class replay(Messenger):

  def __init__(self, fn, guide_trace):
    self.guide_trace = guide_trace
    super(replay, self).__init__(fn)  # self.fn = fn

  def process_msg(self, msg):
    if msg['name'] in self.guide_trace:  # check if site exists in guide's trace
      msg['value'] = self.guide_trace[msg['name']]['value']  # replace value in msg


class block(Messenger):

  def __init__(self, fn=None, hide_fn=lambda msg : True):
    self.hide_fn = hide_fn  # by default, blocks all messages
    super(block, self).__init__(fn)  # self.fn = fn

  def process_msg(self, msg):
    if self.hide_fn(msg):
      msg['stop'] = True


def sample(name, fn, obs=None):
  # IF STACK is empty (no handlers)
  #  just draw a sample; no effects
  if len(STACK) == 0:
    return fn()

  # ELSE encode information from sample statement into a message
  init_msg = {
      'type'  : 'sample',
      'name'  : name,
      'fn'    : fn,
      'args'  : (),
      'value' : obs
      }

  # send message to handlers
  msg = apply_stack(init_msg)  # handlers may override value
  return msg['value']


def apply_stack(msg):  # called by `sample` and `param`
  # iterate through STACK bottom-up (reversed)
  for handler_idx, handler in enumerate(reversed(STACK)):
    # we call "process_msg" when running bottom-up
    handler.process_msg(msg)  # handler modifies msg or reads from msg
    # if there is a block (if STOP flag is set)
    if msg.get('stop'):  # no more operations up the STACK
      break

  # if none of the handlers set the "value" of msg
  #  run msg['fn'] and set it
  if msg['value'] is None:
    msg['value'] = msg['fn'](*msg['args'])  # args :=> empty for `sample`

  # pick up where we stopped; iterate top-down
  for handler in STACK[-handler_idx - 1:]:
    handler.postprocess_msg(msg)

  return msg


def param(name, init_value=None,
  constraint=torch.distributions.constraints.real):

  # a function similar to Distribution.sample()
  # draw a value from PARAM_STORE
  def fn(init_value, constraint):
    if name in PARAM_STORE:  # IF parameter exists in STORE
      unconstrained_value, constraint = PARAM_STORE[name]
    else:  # ELSE
      # `init_value` shouldn't be None
      assert init_value is not None
      with torch.no_grad():  # do not accumulate gradients from these ops
        constrained_value = init_value.detach()  # detach from device
        unconstrained_value = torch.distributions.transform_to(
            constraint).inv(constrained_value)  # inverse transform
      # make it a learnable torch parameter
      unconstrained_value.requires_grad_()  # in-place
      # add unconstrained value and constraint to PARAM_STORE
      PARAM_STORE[name] = unconstrained_value, constraint

    # now return value from PARAM_STORE
    # constrain it
    constrained_value = torch.distributions.transform_to(
        constraint)(unconstrained_value)  # constrain value from PARAM_STORE

    constrained_value.unconstrained = weakref.ref(unconstrained_value)
    return constrained_value

  # if STACK is empty (no handlers)
  #  just draw value from PARAM_STORE and return; no effects
  if len(STACK) == 0:
    return fn(init_value, constraint)

  # ELSE encode information from `param` statement into a msg
  init_msg = {
      'type'  : 'param',
      'name'  : name,
      'fn'    : fn,
      'args'  : (init_value, constraint),
      'value' : None
      }

  # send msg to handlers
  msg = apply_stack(init_msg)  # handlers may override value
  return msg['value']


def generate_data():
  return torch.randn(100) + 9


if __name__ == '__main__':
  # generate data
  data = generate_data()

  def model(data):
    z_loc = sample('z_loc', pdist.Normal(0., 1.))
    # normally distributed observations
    # for datapoint in data:
    sample('obs', pdist.Normal(z_loc, 1.), obs=data)
    # break

  def guide(data):
    # define parameters
    #  loc and scale for latent variable `z_loc`
    guide_loc = param('guide_loc', torch.tensor(0.))
    guide_scale = param('guide_scale', torch.tensor(1.),
        constraint=torch.distributions.constraints.positive
        )  # .exp()
    # we would like to learn the distribution `loc`
    sample('z_loc', pdist.Normal(guide_loc, guide_scale))

  # clear parameter store
  PARAM_STORE.clear()

  # learning rate
  lr = 0.02
  # training steps
  num_steps = 1000

  # SVI for inference
  svi = SVI(model, guide, optimizer=Adam({'lr' : lr}), loss_fn=elbo)

  for step in range(num_steps):
    loss = svi.step(data)
    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  for name, (value, constraint) in PARAM_STORE.items():
    print("{} = {}".format(name, value.detach().cpu().numpy()))

  print('SCALE : ', param('guide_scale'))
  print('LOC : ', param('guide_loc'))
