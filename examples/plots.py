"""Plots

1. ELBO (loss vs step)
2. Parameter (a vs step)
3. Density Plot (y vs x_i)
4. Posterior Predictive Distribution (y vs x_i)

"""

import torch.distributions as tdist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')


def elbo(losses):
  plt.plot(losses)
  plt.title('ELBO')
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.show()


def param(param_history, ground_truth, name='param'):
  plt.plot([0, len(param_history)], [ground_truth, ground_truth], 'k:')
  plt.plot(param_history)
  plt.xlabel('step')
  plt.ylabel(name)
  plt.show()


def sample_normal(mu, sigma, N=100):
  return tdist.Normal(mu, sigma).sample([N])


def density_plot(samples):
  sns.distplot(samples.data.numpy())
  plt.show()


def posterior_predictive(train_data, x, y_pred):
  train_x, train_y = train_data
  plt.plot(x.numpy(), np.mean(y_pred, axis=0), label='Mean Posterior Predictive')
  plt.fill_between(
      x.numpy().reshape(-1),
      np.percentile(y_pred, 0., axis=0),
      np.percentile(y_pred, 99.5, axis=0),
      alpha=0.5, label='Posterior Predictive'
  )
  plt.scatter(train_x[:80], train_y[:80], s=12)
  plt.show()
