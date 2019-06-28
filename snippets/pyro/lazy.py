class LazyValue(object):
    def __init__(self, fn, *args, **kwargs):
        self._expr = (fn, args, kwargs)
        self._value = None

    def __str__(self):
        return "({} {})".format(str(self._expr[0]), " ".join(map(str, self._expr[1])))

    def evaluate(self):
        if self._value is None:
            fn, args, kwargs = self._expr
            fn = fn.evaluate() if isinstance(fn, LazyValue) else fn
            args = tuple(arg.evaluate() if isinstance(arg, LazyValue) else arg
                         for arg in args)
            kwargs = {k: v.evaluate() if isinstance(v, LazyValue) else v
                      for k, v in kwargs.items()}
            self._value = fn(*args, **kwargs)
        return self._value

class LazyMessenger(pyro.poutine.messenger.Messenger):
    def _process_message(self, msg):
        if msg["type"] in ("apply", "sample") and not msg["done"]:
            msg["done"] = True
            msg["value"] = LazyValue(msg["fn"], *msg["args"], **msg["kwargs"])

@effectful(type="apply")
def add(x, y):
    return x + y

@effectful(type="apply")
def mul(x, y):
    return x * y

@effectful(type="apply")
def sigmoid(x):
    return torch.sigmoid(x)

@effectful(type="apply")
def normal(loc, scale):
    return dist.Normal(loc, scale)

def biased_scale(guess):
    weight = pyro.sample("weight", normal(guess, 1.))
    tolerance = pyro.sample("tolerance", normal(0., 0.25))
    return pyro.sample("measurement", normal(add(mul(weight, 0.8), 1.), sigmoid(tolerance)))

with LazyMessenger():
    v = biased_scale(8.5)
    print(v)
    print(v.evaluate())
