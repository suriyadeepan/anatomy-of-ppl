class LogJointMessenger2(poutine.messenger.Messenger):

    def __init__(self, cond_data):
        self.data = cond_data

    def __call__(self, fn):
        def _fn(*args, **kwargs):
            with self:
                fn(*args, **kwargs)
                return self.logp.clone()
        return _fn

    def __enter__(self):
        self.logp = torch.tensor(0.)
        return super(LogJointMessenger2, self).__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.logp = torch.tensor(0.)
        return super(LogJointMessenger2, self).__exit__(exc_type, exc_value, traceback)

    def _pyro_sample(self, msg):
        if msg["name"] in self.data:
            msg["value"] = self.data[msg["name"]]
            msg["done"] = True

    def _pyro_post_sample(self, msg):
        assert msg["done"]  # the "done" flag asserts that no more modifications to value and fn will be performed.
        self.logp = self.logp + (msg["scale"] * msg["fn"].log_prob(msg["value"])).sum()


with LogJointMessenger2(cond_data={"measurement": 9.5, "weight": 8.23}) as m:
    scale(8.5)
    print(m.logp)
