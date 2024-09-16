import numpy as np
from d2pc import *
from msp import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--wdir", type=str, default=None, help="Working directory")
parser.add_argument("-N", type=int, default=20, help="Prediction horizon")
parser.add_argument("--y-lim", type=float, default=.05, help="Y limit (only positive)")
parser.add_argument("--u-lim", type=float, default=2.5, help="U limit (2-sided)")
parser.add_argument("--qc", type=float, default=1, help="Stage cost")
parser.add_argument("--rc", type=float, default=1e-1, help="Control input cost")
parser.add_argument("--p", type=float, default=0.90, help="Chance constraint satisfaction probability")
parser.add_argument("--delta", type=float, default=0.95, help="Confidence level for parametric uncertainty")
parser.add_argument("--bx0", type=float, default=-.2, help="initial state mean")

args = parser.parse_args()
set_seed(args.seed)

if args.wdir is not None:
    dir_name = os.path.join("outputs", "control", args.wdir)
    if not os.path.exists(os.path.join(dir_name, "est.pkl")):
        raise ValueError("Given working directory does not have est-multi.pkl")
else:
    dir_name = get_last_folder('control')
    if dir_name is None:
        raise ValueError("No valid folder found, run est_multi.py first")
    if not os.path.exists(os.path.join(dir_name, "est-multi.pkl")):
        raise ValueError("Given working dirrectory does not have est-multi.pkl, run est-multi.py first")

with open(os.path.join(dir_name, "est-multi.pkl"), "rb") as f:
    data = pickle.load(f)
    true_sys = data["true_sys"]
    ys_all, us_all, syss = data["ys_all"], data["us_all"], data["syss_all"]

# setup
p, delta = args.p, args.delta
epsilon = (1+delta)/2 # equal risk for both sets
Ks = range(1, args.N+1)
costQ = np.eye(true_sys.ny) * args.qc
costR = np.eye(true_sys.nu) * args.rc
bar_x0 = np.ones(true_sys.nx) * args.bx0
Sigma_x0 = true_sys.P0
Hy = np.eye(true_sys.ny) / args.y_lim
Hu = np.eye(true_sys.nu) / args.u_lim
Hu = np.vstack([Hu, -Hu])

us_true = solve_nominal_mpc(true_sys, args.N, bar_x0, Sigma_x0, Hy, Hu, costQ, costR, p)
exp_cost_true, cc_sat_true = compute_metrics(true_sys, us_true, bar_x0, Sigma_x0, Hy, costQ, costR)
msp_est_times = []
exp_costs_prop, cc_sats_prop, off_t_prop = [], [], []
exp_costs_el, cc_sats_el, off_t_el = [], [], []
for i, (sys, ys, us) in tqdm(enumerate(zip(syss, ys_all, us_all)), total=len(syss)):
    # compute msp
    with timer() as t:
        mus, precs = list(zip(*blr(sys, ys, us, Ks, mode="kf")))
        covs = [psd_inverse(prec) for prec in precs]
        msp_est_times.append(t())

    # compute offline tightening terms for proposed solution
    with timer() as t:
        tights_prop = compute_tights_msp_offline(sys, Ks, mus, covs, Sigma_x0, Hy, p, delta, epsilon)
        off_t_prop.append(t())

    try:
        # solve MPC with MSP
        _, us_msp_prop = solve_msp_mpc(args.N, mus, covs, bar_x0, Hy,
                                Hu, costQ, costR, delta, epsilon, tights_prop)
        exp_cost_prop, cc_sat_prop = compute_metrics(true_sys, us_msp_prop, bar_x0, Sigma_x0, Hy, costQ, costR)
        exp_costs_prop.append(exp_cost_prop)
        cc_sats_prop.append(cc_sat_prop)
    except:
        exp_costs_prop.append(np.nan)
        cc_sats_prop.append(np.empty((Hy.shape[1], args.N)))

    # compute offline tightening terms for ellipsoidal method
    with timer() as t:
        tights_el = compute_tights_msp_ellipsoidal_offline(sys, Ks, mus, covs, Sigma_x0, Hy, p, delta)
        off_t_el.append(t())
    try:
        _, us_msp_el = solve_msp_mpc(args.N, mus, covs, bar_x0, Hy, Hu, costQ, costR, delta, epsilon, tights_el, ellipsoidal=True)
        exp_cost_el, cc_sat_el = compute_metrics(true_sys, us_msp_prop, bar_x0, Sigma_x0, Hy, costQ, costR)
        exp_costs_el.append(exp_cost_el)
        cc_sats_el.append(cc_sat_el)
    except:
        exp_costs_el.append(np.nan)
        cc_sats_el.append(np.empty((Hy.shape[1], args.N)))

## save results
exp_costs_prop, exp_costs_el = np.array(exp_costs_prop), np.array(exp_costs_el)
cc_sats_prop, cc_sats_el = np.array(cc_sats_prop), np.array(cc_sats_el)
off_t_prop, off_t_el = np.array(off_t_prop), np.array(off_t_el)

with open(os.path.join(dir_name, "ctrl.pkl"), "wb") as f:
    pickle.dump({"exp_cost_true": exp_cost_true, "cc_sat_true": cc_sat_true, 
                 "exp_costs_prop": exp_costs_prop, 'exp_costs_el':exp_costs_el,
                 "cc_sats_prop": cc_sats_prop, "cc_sats_el": cc_sats_el,
                 "off_t_prop": off_t_prop, "off_t_el": off_t_el,
                 "msp_est_times": msp_est_times}, f)
