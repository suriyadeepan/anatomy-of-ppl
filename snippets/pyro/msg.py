msg = {
    # The following fields contain the name, inputs, function, and output of a site.
    # These are generally the only fields you'll need to think about.
    "name": "x",
    "fn": dist.Bernoulli(0.5),
    "value": None,  # msg["value"] will eventually contain the value returned by pyro.sample
    "is_observed": False,  # because obs=None by default; only used by sample sites
    "args": (),  # positional arguments passed to "fn" when it is called; usually empty for sample sites
    "kwargs": {},  # keyword arguments passed to "fn" when it is called; usually empty for sample sites
    # This field typically contains metadata needed or stored by a particular inference algorithm
    "infer": {"enumerate": "parallel"},
    # The remaining fields are generally only used by Pyro's internals,
    # or for implementing more advanced effects beyond the scope of this tutorial
    "type": "sample",  # label used by Messenger._process_message to dispatch, in this case to _pyro_sample
    "done": False,
    "stop": False,
    "scale": torch.tensor(1.),  # Multiplicative scale factor that can be applied to each site's log_prob
    "mask": None,
    "continuation": None,
    "cond_indep_stack": (),  # Will contain metadata from each pyro.plate enclosing this sample site.
}
