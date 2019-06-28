"""Bayesian Linear Regression"""
import minipyro as pyro

import torch

from torch.distributions import constraints
import pyro.distributions as pdist
import torch.distributions as tdist

import plots
import numpy as np

from random import shuffle


def iris(datafile='./iris.data'):
  # label to index lookup
  # label2idx = { 'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 }
  label2idx = { 'Iris-setosa' : 0, 'Iris-versicolor' : 1 }
  lines = [ l.replace('\n', '').strip() for l in open(datafile).readlines() ]
  # shuffle lines
  shuffle(lines)
  features, labels = [], []
  for line in lines:
    # super-annoying empty last line
    if not line:
      break

    items = line.split(',')
    label = items[-1]

    # check if label is in label2idx
    if label not in label2idx:
      continue

    features.append([ float(i) for i in items[:-1] ])
    labels.append(label2idx[label])

  # train/test separation
  k = int(0.8 * len(features))
  train_x, train_y = features[:k], labels[:k]
  test_x, test_y = features[k:], labels[k:]
  # convenience
  t = torch.tensor

  return (
      ( t(train_x), t(train_y).float() ),
      ( t(test_x), t(test_y).float() )
      )


def model(x, y):
  w = pyro.sample('w', pdist.Normal(torch.zeros(4), torch.ones(4)))
  b = pyro.sample('b', pdist.Normal(0., 1.))

  # define logistic regression model
  y_hat = torch.sigmoid((w * x).sum(dim=1) + b)

  # variance of distribution centered around y
  # sigma = pyro.sample('sigma', pdist.Normal(0., 0.01))

  pyro.sample('obs', pdist.Bernoulli(y_hat), obs=y)


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.zeros(4))
  w_scale = pyro.param('w_scale', torch.ones(4),
      constraint=constraints.positive
      )
  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.tensor(0.))
  b_scale = pyro.param('b_scale', torch.tensor(1.), constraint=constraints.positive)

  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))


def prob_forward(x):
  w = tdist.Normal(pyro.param('w_loc'), pyro.param('w_scale'))
  b = tdist.Normal(pyro.param('b_loc'), pyro.param('b_scale'))
  return torch.sigmoid((w.sample([1]) * x).sum() + b.sample([1]))


def sample_from_posterior(x, fwd, n=100):
  return np.array([ fwd(x).detach().numpy().reshape(-1) for _ in range(n) ])


if __name__ == '__main__':
  # generate data
  (train_x, train_y), (test_x, test_y) = iris()

  # clear parameter store
  pyro.PARAM_STORE.clear()

  # learning rate
  lr = 0.005
  # training steps
  num_steps = 1000

  # SVI for inference
  svi = pyro.SVI(model, guide, optimizer=pyro.Adam({'lr' : lr}), loss_fn=pyro.elbo)

  losses, w, b = [], [], []
  for step in range(num_steps):
    loss = svi.step(train_x, train_y)
    # if step % 100 == 0:
    losses.append(loss)
    w.append(pyro.param('w_loc').data.numpy())
    b.append(pyro.param('b_loc').item())

    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  w_ps = pyro.param('w_loc')
  b_ps = pyro.param('b_loc')

  print('w : {}; b : {}'.format(w_ps, b_ps))

  def predict(x):
    x = torch.tensor(x)
    return torch.sigmoid((w_ps * x).sum() + b_ps)

  correcto = 0
  for xi, yi in zip(test_x, test_y):
    num_samples = 1000
    samples = sample_from_posterior(torch.tensor(xi).view(1, -1),
        prob_forward, num_samples)
    correcto = (((samples > 0.5) == yi.item()).sum())
    print('[{}] {}/{} with {}% certainty'.format(yi, correcto, num_samples,
      100. * correcto / num_samples))
