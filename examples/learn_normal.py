import minipyro as pyro
import torch

from torch.distributions import constraints
import pyro.distributions as pdist

import plots


def generate_data():
  return torch.randn(100) + 3


if __name__ == '__main__':
  # generate data
  data = generate_data()
  # true model
  print('True model : Normal(3., 1.)')

  def model(data):
    z_loc = pyro.sample('z_loc', pdist.Normal(0., 1.))
    # normally distributed observations
    # for datapoint in data:
    pyro.sample('obs', pdist.Normal(z_loc, 1.), obs=data)
    # break

  def guide(data):
    # define parameters
    #  loc and scale for latent variable `z_loc`
    guide_loc = pyro.param('guide_loc', torch.tensor(0.))
    guide_scale = pyro.param('guide_scale', torch.tensor(1.),
        constraint=constraints.positive
        )
    # we would like to learn the distribution `loc`
    pyro.sample('z_loc', pdist.Normal(guide_loc, guide_scale))

  # clear parameter store
  pyro.PARAM_STORE.clear()

  # learning rate
  lr = 0.01
  # training steps
  num_steps = 4000

  # SVI for inference
  svi = pyro.SVI(model, guide, optimizer=pyro.Adam({'lr' : lr}), loss_fn=pyro.elbo)

  losses, z = [], []
  for step in range(num_steps):
    loss = svi.step(data)
    losses.append(loss)
    z.append(pyro.param('guide_loc').item())
    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  print('LOC : ', pyro.param('guide_loc'))
  print('SCALE : ', pyro.param('guide_scale'))

  # plots.elbo(losses)
  plots.param(z, 3., name='z_mean')
  plots.density_plot(
      plots.sample_normal(
        pyro.param('guide_loc').item(),
        pyro.param('guide_scale').item(),
        N=1000
        )
      )
