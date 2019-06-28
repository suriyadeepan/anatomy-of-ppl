class LogJointMessenger(poutine.messenger.Messenger):

    def __init__(self, cond_data):
        self.data = cond_data

    # __call__ is syntactic sugar for using Messengers as higher-order functions.
    # Messenger already defines __call__, but we re-define it here
    # for exposition and to change the return value:
    def __call__(self, fn):
        def _fn(*args, **kwargs):
            with self:
                fn(*args, **kwargs)
                return self.logp.clone()
        return _fn

    def __enter__(self):
        self.logp = torch.tensor(0.)
        # All Messenger subclasses must call the base Messenger.__enter__()
        # in their __enter__ methods
        return super(LogJointMessenger, self).__enter__()

    # __exit__ takes the same arguments in all Python context managers
    def __exit__(self, exc_type, exc_value, traceback):
        self.logp = torch.tensor(0.)
        # All Messenger subclasses must call the base Messenger.__exit__ method
        # in their __exit__ methods.
        return super(LogJointMessenger, self).__exit__(exc_type, exc_value, traceback)

    # _pyro_sample will be called once per pyro.sample site.
    # It takes a dictionary msg containing the name, distribution,
    # observation or sample value, and other metadata from the sample site.
    def _pyro_sample(self, msg):
        assert msg["name"] in self.data
        msg["value"] = self.data[msg["name"]]
        # Since we've observed a value for this site, we set the "is_observed" flag to True
        # This tells any other Messengers not to overwrite msg["value"] with a sample.
        msg["is_observed"] = True
        self.logp = self.logp + (msg["scale"] * msg["fn"].log_prob(msg["value"])).sum()

with LogJointMessenger(cond_data={"measurement": 9.5, "weight": 8.23}) as m:
    scale(8.5)
    print(m.logp.clone())

scale_log_joint = LogJointMessenger(cond_data={"measurement": 9.5, "weight": 8.23})(scale)
print(scale_log_joint(8.5))
