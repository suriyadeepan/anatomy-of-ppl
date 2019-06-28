def monte_carlo_elbo(model, guide, batch, *args, **kwargs):
    # assuming batch is a dictionary, we use poutine.condition to fix values of observed variables
    conditioned_model = poutine.condition(model, data=batch)

    # we'll approximate the expectation in the ELBO with a single sample:
    # first, we run the guide forward unmodified and record values and distributions
    # at each sample site using poutine.trace
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)

    # we use poutine.replay to set the values of latent variables in the model
    # to the values sampled above by our guide, and use poutine.trace
    # to record the distributions that appear at each sample site in in the model
    model_trace = poutine.trace(
        poutine.replay(conditioned_model, trace=guide_trace)
    ).get_trace(*args, **kwargs)

    elbo = 0.
    for name, node in model_trace.nodes.items():
        if node["type"] == "sample":
            elbo = elbo + node["fn"].log_prob(node["value"]).sum()
            if not node["is_observed"]:
                elbo = elbo - guide_trace.nodes[name]["fn"].log_prob(node["value"]).sum()
    return -elbo


def train(model, guide, data):
    optimizer = pyro.optim.Adam({})
    for batch in data:
        # this poutine.trace will record all of the parameters that appear in the model and guide
        # during the execution of monte_carlo_elbo
        with poutine.trace() as param_capture:
            # we use poutine.block here so that only parameters appear in the trace above
            with poutine.block(hide_fn=lambda node: node["type"] != "param"):
                loss = monte_carlo_elbo(model, guide, batch)

        loss.backward()
        params = set(node["value"].unconstrained()
                     for node in param_capture.trace.nodes.values())
        optimizer.step(params)
        pyro.infer.util.zero_grads(params)
