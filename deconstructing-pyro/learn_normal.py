import minipyro as pyro
import torch

from torch.distributions import constraints
import pyro.distributions as pdist


def generate_data():
  return torch.randn(100) + 9


if __name__ == '__main__':
  # generate data
  data = generate_data()

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

  for step in range(num_steps):
    loss = svi.step(data)
    if step % 100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  print('SCALE : ', pyro.param('guide_scale'))
  print('LOC : ', pyro.param('guide_loc'))
