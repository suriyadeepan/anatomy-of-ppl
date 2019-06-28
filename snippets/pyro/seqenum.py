def sequential_discrete_marginal(model, data, site_name="_RETURN"):

    from six.moves import queue  # queue data structures
    q = queue.Queue()  # Instantiate a first-in first-out queue
    q.put(poutine.Trace())  # seed the queue with an empty trace

    # as before, we fix the values of observed random variables with poutine.condition
    # assuming data is a dictionary whose keys are names of sample sites in model
    conditioned_model = poutine.condition(model, data=data)

    # we wrap the conditioned model in a poutine.queue,
    # which repeatedly pushes and pops partially completed executions from a Queue()
    # to perform breadth-first enumeration over the set of values of all discrete sample sites in model
    enum_model = poutine.queue(conditioned_model, queue=q)

    # actually perform the enumeration by repeatedly tracing enum_model
    # and accumulate samples and trace log-probabilities for postprocessing
    samples, log_weights = [], []
    while not q.empty():
        trace = poutine.trace(enum_model).get_trace()
        samples.append(trace.nodes[site_name]["value"])
        log_weights.append(trace.log_prob_sum())

    # we take the samples and log-joints and turn them into a histogram:
    samples = torch.stack(samples, 0)
    log_weights = torch.stack(log_weights, 0)
    log_weights = log_weights - dist.util.logsumexp(log_weights, dim=0)
    return dist.Empirical(samples, log_weights)
