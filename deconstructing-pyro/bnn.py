"""Bayesian Neural Network"""
import minipyro as pyro
import torch

from torch.distributions import constraints
import pyro.distributions as pdist
import torch.distributions as tdist

import plots
import numpy as np


def random_sample(t, k):
  x, y = t
  indices = torch.randperm(len(x))
  return x[indices][:k], y[indices][:k]


def toy_parabolic_fn(x):
    return x**2 + x + 3.2


def toy_parabolic_data(n):  # TODO : use `n`
  # x = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).reshape(-1, 1)
  # x = torch.tensor([0, 0.5, 0.7, 0.8, 0.9, 1]).reshape(-1, 1)
  # x = torch.arange(0., 1., 0.2).reshape(-1, 1)
  x = torch.tensor([0., 0.4, 0.6, 1.]).reshape(-1, 1)
  y = toy_parabolic_fn(x)
  return x, y


def model(x, y):
  w1 = pyro.sample('w1', pdist.Normal(0., 1.))
  b1 = pyro.sample('b1', pdist.Normal(0., 1.))
  w2 = pyro.sample('w2', pdist.Normal(0., 1.))
  b2 = pyro.sample('b2', pdist.Normal(0., 1.))
  # define model
  mean = w2 * torch.tanh(w1 * x + b1) + b2
  pyro.sample('obs', pdist.Normal(mean, 0.01), obs=y)
  return mean


def guide(x, y):
  # parameters of (w : weight)
  w_loc_1 = pyro.param('w_loc_1', torch.tensor(0.))
  w_scale_1 = pyro.param('w_scale_1', torch.tensor(1.),
      constraint=constraints.positive
      )
  # parameters of (b : bias)
  b_loc_1 = pyro.param('b_loc_1', torch.tensor(0.))
  b_scale_1 = pyro.param('b_scale_1', torch.tensor(1.), constraint=constraints.positive)

  w_loc_2 = pyro.param('w_loc_2', torch.tensor(0.))
  w_scale_2 = pyro.param('w_scale_2', torch.tensor(1.),
      constraint=constraints.positive
      )
  b_loc_2 = pyro.param('b_loc_2', torch.tensor(0.))
  b_scale_2 = pyro.param('b_scale_2', torch.tensor(1.), constraint=constraints.positive)

  w1 = pyro.sample('w1', pdist.Normal(w_loc_1, w_scale_1))
  b1 = pyro.sample('b1', pdist.Normal(b_loc_1, b_scale_1))
  w2 = pyro.sample('w2', pdist.Normal(w_loc_2, w_scale_2))
  b2 = pyro.sample('b2', pdist.Normal(b_loc_2, b_scale_2))


def prob_forward(x):
  w1 = tdist.Normal(pyro.param('w_loc_1'), pyro.param('w_scale_1'))
  b1 = tdist.Normal(pyro.param('b_loc_1'), pyro.param('b_scale_1'))
  w2 = tdist.Normal(pyro.param('w_loc_2'), pyro.param('w_scale_2'))
  b2 = tdist.Normal(pyro.param('b_loc_2'), pyro.param('b_scale_2'))
  h1 = torch.tanh(w1.sample(x.size()) * x + b1.sample(x.size()))
  return w2.sample(h1.size()) * h1 + b2.sample(h1.size())


def sample_from_posterior(x, fwd, n=100):
  return np.array([ fwd(x).detach().numpy().reshape(-1) for _ in range(n) ])


if __name__ == '__main__':
  # generate data
  x, y = toy_parabolic_data(10)
  # num of data points
  print('len(trainset) : ', len(x))
  print(x, y)

  # clear parameter store
  pyro.PARAM_STORE.clear()

  # learning rate
  lr = 0.008
  # training steps
  num_steps = 2000

  # SVI for inference
  svi = pyro.SVI(model, guide, optimizer=pyro.Adam({'lr' : lr}), loss_fn=pyro.elbo)

  losses, w, b = [], [], []
  for step in range(num_steps):
    loss = svi.step(x, y)
    losses.append(loss)
    # w.append(pyro.param('w_loc').item())
    # b.append(pyro.param('b_loc').item())

    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  # print('W(LOC) : ', pyro.param('w_loc').item(), pyro.param('w_scale').item())
  # print('b(LOC) : ', pyro.param('b_loc').item(), pyro.param('b_scale').item())

  # --- PLOTS ---
  # plots.elbo(losses)
  # plots.param(w, 2., name='w')
  # plots.param(b, 1., name='b')
  ###############
  """
  plots.density_plot(
      plots.sample_normal(
        pyro.param('w_loc').item(),
        pyro.param('w_scale').item(),
        N=1000
        )
      )
  """

  x_pred = torch.linspace(-3, 3, 1000).reshape(-1, 1)
  plots.posterior_predictive((x, y), x_pred,
      sample_from_posterior(x_pred, prob_forward)
      )
