#%%

import numpy
import matplotlib.pyplot as plt
import pystan

a = 1.5
b = 0.5

t = numpy.linspace(0.0, 1.5 * numpy.pi / 2.0, 15)
y1 = a * t * numpy.sin(t)
y2 = a * t * numpy.sin(t) + b * numpy.random.randn(15)

plt.plot(t, y1, 'b')
plt.plot(t, y2, 'r*')
plt.show()

t = t[1:]
y0 = y2[0]
y2 = y2[1:]

#%%
model_code = """
functions {
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[1];
    dydt[1] <- theta[1] * t * cos(t) + theta[1] * sin(t);
    return dydt;
  }
}

data {
  int<lower=1> T;
  real y[T];
  real t0;
  real ts[T];
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real y0[1];
  real<lower=0.0> sigma;
  real theta[1];
}

model {
  real y_hat[T, 1];
  sigma ~ cauchy(0, 2.5);
  theta ~ normal(0.0, 10.0);
  y0 ~ normal(0, 1);

  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
  for (t in 1:T)
    y[t] ~ normal(y_hat[t], sigma);
}

generated quantities {
  real y_hat[T, 1];
  real y_hat_err[T];

  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
  // add measurement error
  for (t in 1:T) {
    y_hat[t,1] <- y_hat[t,1];
    y_hat_err[t] <- normal_rng(0, sigma);
  }
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
    'T' : len(y2),
    'y' : y2,
    't0' : 0.0,
    'ts' : t
})

print fit

#%%

y0s = fit.extract()['y0'][2000:]
y2s = fit.extract()['y_hat'][2000:]
yerrs = fit.extract()['y_hat_err'][2000:]

plt.plot(numpy.concatenate(([0.0], t)), numpy.concatenate(([y0], y2)), '*')

for i in numpy.random.choice(range(2000), 20):
    xs = numpy.concatenate(([0.0], t))
    ys = numpy.concatenate(([y0s[i]], y2s[i, :, 0]))
    plt.plot(xs, ys)

plt.show()