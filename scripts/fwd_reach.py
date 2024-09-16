import numpy as np
from d2pc import *
from msp import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num-masses", type=int, default=3, help="Number of masses")
parser.add_argument("--num-actuated", type=int, default=1, help="Number of actuated masses (starts from last)")
parser.add_argument("-T", type=int, default=1000, help="Data length")
parser.add_argument("-N", type=int, default=20, help="Prediction horizon")
parser.add_argument("-q", type=float, default=1e-3, help="Disturbance covariance")
parser.add_argument("-r", type=float, default=1e-3, help="Measurement noise covariance")
parser.add_argument("-p0", type=float, default=1e-3, help="Initial state covariance")
parser.add_argument("-u", type=float, default=5, help="Excitation input magnitude")
parser.add_argument("-p", type=float, default=.9, help="Chance constraint satisfaction probability")
parser.add_argument("-delta", type=float, default=.95, help="Confidence level for parametric uncertainty")
args = parser.parse_args()
set_seed(args.seed)

str_time = datetime.now().strftime('%Y%m%d%H%M%S')
folder = os.path.join("outputs", "fwd-reach", str_time)
os.makedirs(folder, exist_ok=True)
logger = get_logger(os.path.join(folder, "info"))

# construct true system
n_m, n_a = args.num_masses, args.num_actuated
true_sys = construct_spring_mass_sys(n_m, n_a, args.q, args.r)

# simulate true system
_, ys, us, ws, vs = true_sys.simulate(
    args.T, RandomNormalController(true_sys.nu, sigma_u=2), return_noises=True)

# estimate system
with timer() as t:
    Q0 = np.eye(n_m) * args.q * np.random.rand()
    R0 = np.eye(n_m) * args.r * np.random.rand()
    P00 = np.eye(n_m*2) * args.r * np.random.rand()
    mu00 = np.random.rand(n_m*2) * 1e-1
    est_sys, E = estimate_arx_sys_canon(ys, us, order=2, Q=Q0, Qtype="scaled", R=R0, Rtype="scaled",
                                        mu0=mu00, P0=P00, ignore_Theta=True,
                                        update_init_dist=True, verbose=0, max_iter=2500)
    est_sys = convert_canon_to_IO(est_sys)
    est_sys.P0 = est_sys.get_steady_post_covar()
    logger.info("Estimation time for state-space model: " + f"{t():.3f}"+"s")

# compute MSP params
Ks = range(1, args.N+1)
with timer() as t:
    mus, precs = list(zip(*blr(est_sys, ys, us, Ks, mode="kf")))
    logger.info("Estimation time for multi-step models: " + f"{t():.3f}"+"s")

# initial state distribution
bar_x0, Sigma_x0 = true_sys.mu0, true_sys.P0

# compute error bounds
h = np.array([0, 0, 1])
p, delta = args.p, args.delta
Ks = range(1, len(mus)+1)
u_ref = np.ones((len(mus), true_sys.nu)) * args.u
covs = [psd_inverse(prec) for prec in precs]
mean_est, tights_m_est, tights_c_est = compute_tights_msp(est_sys, Ks, mus, covs, bar_x0,
                                                          Sigma_x0, u_ref, h, p, delta, epsilon=None, two_sided=True)
tights_est = tights_m_est + tights_c_est
mean_el, tights_m_el, tights_c_el = compute_tights_msp_ellipsoidal(est_sys, Ks, mus, covs, bar_x0, Sigma_x0, u_ref, h, p, delta, two_sided=True)
tights_el = tights_m_el + tights_c_el
mean_sampling, tights_sampling = ct_sampling(est_sys, Ks, mus, covs,
                                             bar_x0, Sigma_x0, u_ref, h, p, delta, two_sided=True)

# compute D2PC controller and tightening
costQ, costR = np.eye(true_sys.nx), np.eye(true_sys.nu)
h_ = np.array([np.hstack([h@est_sys.C, np.zeros(true_sys.nu)])])
J = np.eye(true_sys.ny * (true_sys.nx + true_sys.nu))
ctrlr = d2pc_pipeline_open_loop(est_sys, ys, us, J, np.zeros(J.shape[1]), p, delta, costQ, 
                                costR, h_, Sigma_x0, T_err=10, logger=None)
mean_d2pc, tights_m_d2pc, tights_c_d2pc = compute_tight_d2pc(
    ctrlr, u_ref, p, h, bar_x0, two_sided=True)
tights_d2pc = tights_m_d2pc + tights_c_d2pc

# plot
means = [mean_d2pc, mean_el, mean_est, mean_sampling]
tights = [tights_d2pc, tights_el, tights_est, tights_sampling]
names = ["Sequential", "Ellipsoidal (Prop. 2)", "Proposed (Thm. 1)", "Sampling"]
colors = ["tab:orange", "tab:red", "tab:blue", "tab:green"]
fig = plot_err_two_sided(means, tights, names, colors, lw=2, alpha=0.6)
fig.savefig(os.path.join(folder, "fwd-reach.png"))
