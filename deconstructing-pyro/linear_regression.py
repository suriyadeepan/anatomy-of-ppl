"""Bayesian Linear Regression"""
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


def noisy():
  x = torch.arange(-0.05, 0.2, 0.1)
  w, b = 2., 1.
  y = w * x + b + torch.normal(torch.tensor(0.), torch.tensor(0.01))
  print('TRUE DISTRIBUTION : {w}x + {b}'.format(w=w, b=b))
  x, y = random_sample((x, y), len(x))
  k = int(len(x) * 0.8)

  return (x[:k], y[:k]), (x[k:], y[k:])


def model(x, y):
  w = pyro.sample('w', pdist.Normal(0., 1.))
  b = pyro.sample('b', pdist.Normal(0.5, 1.))
  # define model
  mean = w * x + b
  # mean = w * x + 0.3
  # variance of distribution centered around y
  # sigma = pyro.sample('sigma', pdist.Normal(0., 0.01))
  pyro.sample('obs', pdist.Normal(mean, 0.01), obs=y)
  return mean


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.tensor(1.))
  w_scale = pyro.param('w_scale', torch.tensor(1.),
      constraint=constraints.positive
      )
  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.tensor(0.))
  b_scale = pyro.param('b_scale', torch.tensor(1.), constraint=constraints.positive)
  # parameters of (sigma)
  # sigma_loc = pyro.param('sigma_loc', torch.tensor(0.), constraint=constraints.positive)  # .exp()

  # sample (w, b, sigma)
  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))
  # sigma = pyro.sample('sigma', pdist.Normal(sigma_loc, torch.tensor(0.05)))


def prob_forward(x):
  w = tdist.Normal(pyro.param('w_loc'), pyro.param('w_scale'))
  b = tdist.Normal(pyro.param('b_loc'), pyro.param('b_scale'))
  return w.sample(x.size()) * x + b.sample(x.size())


def sample_from_posterior(x, fwd, n=100):
  return np.array([ fwd(x).detach().numpy().reshape(-1) for _ in range(n) ])


if __name__ == '__main__':
  # generate data
  (train_x, train_y), (test_x, test_y) = noisy()
  # num of data points
  print('len(trainset) : ', len(train_x))

  # clear parameter store
  pyro.PARAM_STORE.clear()

  # learning rate
  lr = 0.005
  # training steps
  num_steps = 5000

  # SVI for inference
  svi = pyro.SVI(model, guide, optimizer=pyro.Adam({'lr' : lr}), loss_fn=pyro.elbo)

  losses, w, b = [], [], []
  for step in range(num_steps):
    loss = svi.step(train_x, train_y)
    # if step % 100 == 0:
    losses.append(loss)
    w.append(pyro.param('w_loc').item())
    b.append(pyro.param('b_loc').item())

    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  print('W(LOC) : ', pyro.param('w_loc').item(), pyro.param('w_scale').item())
  print('b(LOC) : ', pyro.param('b_loc').item(), pyro.param('b_scale').item())
  # print('sigma(LOC) : ', pyro.param('sigma_loc').item())

  # plots.elbo(losses)
  # plots.param(w, 2., name='w')
  # plots.param(b, 1., name='b')
  """
  plots.density_plot(
      plots.sample_normal(
        pyro.param('w_loc').item(),
        pyro.param('w_scale').item(),
        N=1000
        )
      )
  """

  x_pred = torch.linspace(-2, 2, 100).reshape(-1, 1)
  plots.posterior_predictive((train_x, train_y), x_pred,
      sample_from_posterior(x_pred, prob_forward)
      )
