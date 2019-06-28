import torch

STACK = []
PARAM_STORE = {}


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
    if msg['stop']:  # no more operations up the STACK
      break

  # if none of the handlers set the "value" of msg
  #  run msg['fn'] and set it
  if not msg['value']:
    msg['value'] = msg['fn'](*msg['args'])  # args :=> empty for `sample`

  # pick up where we stopped; iterate top-down
  for handler in STACK[-handler_idx - 1:]:
    handler.postprocess_message()

  return msg


def param(name, init_value=None, constraint=torch.distributions.constraints.real):

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
      PARAM_STORE[name] = (unconstrained_value, constraint)

    # now return value from PARAM_STORE
    # constrain it
    constrained_value = torch.distributions.transform_to(
        constraint)(unconstrained_value)  # constrain value from PARAM_STORE
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
