import numpy as np
from d2pc import *
from msp import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n-sim", type=int,
                    default=1000, help="Number of simulations")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num-masses", type=int, default=3, help="Number of masses")
parser.add_argument("--num-actuated", type=int, default=1, help="Number of actuated masses (starts from last)")
parser.add_argument("-T", type=int, default=1000, help="Data length")
parser.add_argument("-N", type=int, default=20, help="Prediction horizon")
parser.add_argument("-q", type=float, default=1e-3, help="Disturbance covariance")
parser.add_argument("-r", type=float, default=1e-3, help="Measurement noise covariance")
args = parser.parse_args()
set_seed(args.seed)

str_time = datetime.now().strftime('%Y%m%d%H%M%S')
folder = os.path.join("outputs", "control", str_time)
os.makedirs(folder, exist_ok=True)
logger = get_logger(os.path.join(folder, "info-est"))

n_m, n_a = args.num_masses, args.num_actuated
true_sys = construct_spring_mass_sys(n_m, n_a, args.q, args.r)
    
ys_all, us_all, syss_all, times = [], [], [], []
with tqdm(total=args.n_sim) as pbar:
    while len(syss_all) < args.n_sim:
        try:
            # simulate true system
            _, ys, us = true_sys.simulate(
                args.T, RandomNormalController(true_sys.nu, sigma_u=2))

            Q0 = np.eye(n_m) * args.q * np.random.rand()
            R0 = np.eye(n_m) * args.r * np.random.rand()
            P00 = np.eye(n_m*2) * args.r * np.random.rand()
            mu00 = np.random.rand(n_m*2) * 1e-1
            with timer() as t:
                est_sys, E = estimate_arx_sys_canon(ys, us, order=2, Q=Q0, Qtype="scaled", R=R0, Rtype="scaled",
                                                    mu0=mu00, P0=P00, ignore_Theta=True,
                                                    update_init_dist=True, verbose=0, max_iter=2500)
                times.append(t())
            est_sys = convert_canon_to_IO(est_sys)
            est_sys.P0 = est_sys.get_steady_post_covar()
            ys_all.append(ys)
            us_all.append(us)
            syss_all.append(est_sys)
            pbar.update(1)
        except:
            logger.info("Error in estimation, skipping this instance")
            continue

with open(os.path.join(folder, "est-multi.pkl"), "wb") as f:
    pickle.dump({"true_sys": true_sys, "ys_all": ys_all, 
                "us_all":us_all, "syss_all":syss_all, "times": times}, f)