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
bar_x0 = np.ones(true_sys.nx) * args.bx0
Sigma_x0 = true_sys.P0
Hy = np.eye(true_sys.ny) / args.y_lim
Hu = np.eye(true_sys.nu) / args.u_lim
Hu = np.vstack([Hu, -Hu])

h_ = block_diag(Hy@true_sys.C, Hu)
J = np.eye(true_sys.ny * (true_sys.nx + true_sys.nu))
costQ = true_sys.C.T @ np.eye(true_sys.ny) @ true_sys.C * args.qc
costR = np.eye(true_sys.nu) * args.rc

feas, times = [], []
for i, (sys, ys, us) in tqdm(enumerate(zip(syss, ys_all, us_all)), total=len(syss)):
    try:
        with timer() as t:
            ctrlr = d2pc_pipeline_open_loop(sys, ys, us, J, np.zeros(J.shape[1]), p, delta,
                                    costQ, costR, h_, Sigma_x0, T_err=10, logger=None)
            ctrlr.setup(bar_x0, args.N, ignore_terminal=True)
            times.append(t())
        ctrlr.compute_input_sequence(args.N)
        feas.append(True)
    except:
        feas.append(False)

with open(os.path.join(dir_name, "ctrl-d2pc.pkl"), "wb") as f:
    pickle.dump({"feas": feas, "times": times}, f)
