from .explain import partial_dependence_tau_mp
import numpy as np
import matplotlib.pyplot as plt


def test_partial_dependence_tau_mp():
    size = 100
    inputs = np.zeros((size, 8))
    # temperature
    inputs[:, 0] = np.random.normal(loc=273, scale=5, size=size)
    # rho (density)
    inputs[:, 1] = np.random.uniform(0.1, 1, size)
    # qcin
    inputs[:, 2] = 10 ** np.random.normal(loc=-6, scale=2, size=size)
    # ncin
    inputs[:, 3] = 10 ** np.random.normal(loc=1, scale=1, size=size)
    # qrin
    inputs[:, 4] = 10 ** np.random.normal(loc=-3, scale=1, size=size)
    # nrin
    inputs[:, 5] = 10 ** np.random.normal(loc=1, scale=1, size=size)
    # lcldm
    inputs[:, 6] = np.random.uniform(0.1, 1, size)
    # precip_frac
    inputs[:, 7] = np.random.uniform(0.1, 1, size)


    n_procs = 1
    var_val_count = 10
    tau_outputs = 4
    pd_vals, var_vals = partial_dependence_tau_mp(inputs, var_val_count, n_procs)
    assert pd_vals.shape[0] == inputs.shape[1]
    assert pd_vals.shape[1] == var_val_count
    assert pd_vals.shape[2] == tau_outputs
    assert pd_vals.shape[3] == inputs.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    inputs = ["temp", "rho", "qcin", "ncin", "qrin", "nrin", "lcldm", "precip_frac"]
    for a, ax in enumerate(axes.ravel()):
        ax.plot(var_vals[a], pd_vals[a, :, 3].mean(axis=-1))
        ax.set_title(inputs[a])
    plt.savefig("/tmp/pdp_tau.png")
    return