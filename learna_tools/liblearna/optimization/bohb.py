import os

os.environ["OMP_NUM_THREADS"] = "1"

import logging

logging.basicConfig(level=logging.ERROR)

import argparse
import pickle
import time
import json

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.core.base_iteration import  Datum
from hpbandster.core.result import Result
from hpbandster.optimizers import BOHB as BOHB

from src.optimization.learna_worker import LearnaWorker
from src.optimization.meta_learna_worker import MetaLearnaWorker
from src.optimization.meta_learna_adapt_worker import MetaLearnaAdaptWorker


def load_valid_results(directory):
    data = {}
    time_ref = float('inf')
    budget_set = set()

    with open(os.path.join(directory, 'configs.json')) as fh:
        for line in fh:

            line = json.loads(line)

            if len(line) == 3:
                config_id, config, config_info = line
            if len(line) == 2:
                config_id, config, = line
                config_info = 'N/A'
            data[tuple(config_id)] = Datum(config=config, config_info=config_info)


    exceptions = []
    with open(os.path.join(directory, 'results.json')) as fh:
        for line in fh:
            config_id, budget,time_stamps, result, exception = json.loads(line)

            if exception:
                print("found exception: \n" + f"{exception}" + f"in config: {config_id}")
                exceptions.append(config_id)
                continue


            id = tuple(config_id)

            data[id].time_stamps[budget] = time_stamps
            data[id].results[budget] = result
            data[id].exceptions[budget] = exception

            budget_set.add(budget)
            time_ref = min(time_ref, time_stamps['submitted'])

    for id in exceptions:
        id = tuple(id)
        del data[id]


    # infer the hyperband configuration from the data
    budget_list = sorted(list(budget_set))

    HB_config = {
                        'eta'        : None if len(budget_list) < 2 else budget_list[1]/budget_list[0],
                        'min_budget' : min(budget_set),
                        'max_budget' : max(budget_set),
                        'budgets'    : budget_list,
                        'max_SH_iter': len(budget_set),
                        'time_ref'   : time_ref
                }
    return(Result([data], HB_config))


parser = argparse.ArgumentParser(
    description="Optimization pipeline for Learning to Design RNA"
)
parser.add_argument(
    "--min_budget",
    type=float,
    help="Minimum budget used during the optimization.",
    default=10,
)
parser.add_argument(
    "--max_budget",
    type=float,
    help="Maximum budget used during the optimization.",
    default=10,
)
parser.add_argument(
    "--n_iterations",
    type=int,
    help="Number of iterations performed by the optimizer",
    default=16,
)

parser.add_argument(
    "--n_cores", type=int, help="Number of workers to run in parallel.", default=10
)
parser.add_argument(
    "--worker", help="Flag to turn this into a worker process", action="store_true"
)

parser.add_argument(
    "--run_id",
    type=str,
    help="A unique run id for this optimization run."
    "An easy option is to use the job id of the clusters scheduler.",
)
parser.add_argument("--data_dir", type=str, help="path where the datasets are stored")
parser.add_argument(
    "--nic_name", type=str, help="Which network interface to use for communication."
)
parser.add_argument(
    "--shared_directory",
    type=str,
    help="A directory that is accessible for all processes, e.g. a NFS share.",
)

parser.add_argument("--mode", choices=["learna", "meta_learna", "meta_learna_adapt"], default="learna")
# parser.add_argument(
#     "--load_existing", help="Flag to load exsting data", action="store_true"
# )
# parser.add_argument(
#     "--previous_run_dir",
#     type=str,
#     help="The directory of a previous run to load the data from.",
# )


# args=parser.parse_args("--run_id test --nic_name lo --shared_directory /tmp --n_cores 4 --data_dir src/data --mode L2DesignRNA".split())
args = parser.parse_args()

os.makedirs(args.shared_directory, exist_ok=True)

if args.mode == "learna":
    worker_cls = LearnaWorker
    worker_args = dict(
        data_dir=args.data_dir, num_cores=args.n_cores, train_sequences=range(1, 101)
    )

if args.mode == "meta_learna":
    worker_cls = MetaLearnaWorker
    worker_args = dict(
        data_dir=args.data_dir,
        num_cores=args.n_cores,
        train_sequences=range(1, 100),
        validation_timeout=1,
    )

if args.mode == "meta_learna_adapt":
    worker_cls = MetaLearnaAdaptWorker
    worker_args = dict(
        data_dir=args.data_dir,
        num_cores=args.n_cores,
        train_sequences=range(1, 100000),
        validation_timeout=60,
    )


# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)

if args.worker:
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    w = worker_cls(**worker_args, run_id=args.run_id, host=host)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)
result_logger = hpres.json_result_logger(args.shared_directory, overwrite=True)


# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port
NS = hpns.NameServer(
    run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory
)
ns_host, ns_port = NS.start()

# Most optimizers are so computationally inexpensive that we can affort to run a
# worker in parallel to it. Note that this one has to run in the background to
# not plock!
w = worker_cls(
    **worker_args,
    run_id=args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port
)
w.run(background=True)

print(worker_cls.get_configspace())

print("############# LOADING PREVIOUS RUN")
# previous_run = load_valid_results(args.previous_run_dir)

# Run an optimizer
# We now have to specify the host, and the nameserver information
bohb = BOHB(
    configspace=worker_cls.get_configspace(),
    run_id=args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    min_budget=args.min_budget,
    max_budget=args.max_budget,
    result_logger=result_logger,
    ping_interval=600,
    working_directory=args.shared_directory,
    # previous_result = previous_run,
)
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=1)


# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
with open(os.path.join(args.shared_directory, "results.pkl"), "wb") as fh:
    pickle.dump(res, fh)


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
