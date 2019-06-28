from collections import OrderedDict

STACK = []


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

  def process_message(self, msg):
    pass

  def postprocess_message(self, msg):
    pass


class trace(Messenger):

  def __enter__(self):
    super(trace, self).__enter__()  # execute parent; add to STACK
    self.trace = OrderedDict()  # init trace
    return self.trace  # empty trace

  def postprocess_message(self, msg):
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

  def process_message(self, msg):
    if self.hide_fn(msg):
      msg['stop'] = True
