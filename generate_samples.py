#%%

import numpy
import matplotlib.pyplot as plt
import pystan

a = 1.5
b = 0.5
c = 0.7

t = numpy.linspace(0.0, 1.5 * numpy.pi / 2.0, 15)
y1 = a * t * (c + numpy.sin(t))
y2 = a * t * (c + numpy.sin(t)) + b * numpy.random.randn(15)

plt.plot(t, y1, 'b')
plt.plot(t, y2, 'r*')
plt.show()

t = t[1:]
y0 = y2[0]
y2 = y2[1:]

#%%
model_code = """
functions {
  // This is the 'f' in the ODE y' = f
  //    theta is an array of parameters
  //    x_r is an array of fixed real data (interpolation table n' stuff)
  //    x_i is an array of fixed integer data (dunno if you use this)
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    //Declare the variable we gotta write out
    real dydt[1];

    // Relabel the variables
    real a;
    real c;

    a <- theta[1];
    c <- x_r[1];

    dydt[1] <- a * t * cos(t) + a * (c + sin(t));
    return dydt;
  }
}

// The data block basically contains all the data we're going to pass into Stan
data {
  int<lower=1> T;
  real y[T];
  real t0;
  real a;
  real b;
  real c;
  real y0[1];
  real ts[T];
}

// This is a block useful for rescaling things... I think it just runs once
//    when you start the Stan fit. In this case we fill up the arrays
//    theta and x_r with the data we pass in (so we don't have to do this transformation
//    every time we want to call integrate_ode).
transformed data {
  real theta[1];
  real x_r[1];
  int x_i[0];

  theta[1] <- a;
  x_r[1] <- c;
}

// These are some bullshit parameters we're gonna fit. We're only actually generating
//     data in this Stan run but for some reason it's not letting me use the program without
//     doing a fit. (Figure 19.2 implies this is possible but it wasn't working for me)
parameters {
    real mu;
    real sigma;
}

// Fit our fake model
//    More on the model block later, but this code runs at least once for every
//    sample generated by Stan. It is a likelihood * prior. For this example,
//    it doesn't really matter. We're just fitting the parameters mu and sigma
//    to the data y. It's basically meaningless.
model {
    y ~ normal(mu, sigma);
}

// This is the workhorse. We're gonna tell our ODE to run and add a little noise to it
// This is what you want to get working.

// If you can pass in a ton of parameters and get Stan to execute your ODE model
//    and produce output that corresponds to the output you get with your own ODE
//    solver, that should be the bulk of the work we need to do to set up the model
generated quantities {
  real y_hat[T, 1];

  // Run our ODE! Look in the Stan manual for information on the format of
  //    all this stuff.
  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
  // Add measurement error, this isn't really necessary for your case. We just
  // need to get the ODE running and that's good enough
  for (t in 1:T) {
    y_hat[t,1] <- y_hat[t,1] + b * normal_rng(0, 1.0);
  }
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
    'T' : len(y2),
    'y' : y2,
    't0' : 0.0,
    'a' : a,
    'b' : b,
    'c' : c,
    'y0' : [y0],
    'ts' : t
}, iter = 10) # We're asking Stan to only generate 10 output samples. We end up with 20 I dunno why...

print fit

#%%

# This is how you get data out of Stan
#   Check 'fit.extract().keys()' to see all the variables that were sampled
y2s = fit.extract()['y_hat'][-10:] # Take only the last 10 samples... Dunno why we get 20.

for i in numpy.random.choice(range(len(y2s)), 20):
    xs = numpy.concatenate(([0.0], t))
    ys = numpy.concatenate(([y0], y2s[i, :, 0]))
    plt.plot(xs, ys, '*')

plt.title('Samples (training data as dots')
plt.show()