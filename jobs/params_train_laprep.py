import subprocess
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--exp_name', type=str, default="EXP1")
parser.add_argument('--config_name', type=str, default="al.yaml")
# parser.add_argument('--batch_size', type=int, default=1024)
# parser.add_argument('--seed', type=int, default=1234)


batch_sizes = [512, 1024, 2048]
env_names = ["GridRoom-16"]

args = parser.parse_args()

for env_name in env_names:

    for s in [1234]:

        for bs in batch_sizes:

            job_name = f'{s}-{env_name}-{bs}'

            cmd = [
                "sbatch",
                f"--job-name={job_name}",
                "./run_train_laprep.sh",
                args.exp_name,
                args.config_name,
                str(bs),
                str(s),
                env_name
            ]

            subprocess.run(cmd, check=True)
