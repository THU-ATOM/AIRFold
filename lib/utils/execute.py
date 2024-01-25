import os
import shutil
import subprocess


def execute(
    command,
    verbose: bool = False,
    print_off: bool = False,
    bash: bool = False,
    timeout: float = None,
    log_path: str = None
):
    if bash:
        command = f'bash -c "{command}"'
    if verbose:
        print(command)
        print()
    try:
        if log_path is not None:
            out = open(log_path, 'a')
        elif print_off:
            out = open(os.devnull, 'w')
        else:
            out = None
        p = subprocess.Popen(command, shell=True, stdout=out, stderr=out)
        if timeout is not None:
            print(f"Waiting {timeout}s to kill.")
        p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise  # resume the TimeoutExpired exception
    except KeyboardInterrupt:
        p.kill()
        raise  # resume the KeyboardInterrupt
    return command


def rlaunch_exists():
    return shutil.which('rlaunch') is not None


def rlaunch_wrapper(
    command: str,
    cpu: int = 4,
    gpu: int = 1,
    memory: int = 102400,
    private_machine: str = "group",
    charged_group: str = "health",
    max_wait_time: int = 99999999,
    preemptible: str = "in-replica",
    log_path: str = None,
) -> str:
    """Use rlaunch to run a comand.

    Parameters
    ----------
    command : str
        The command to be executed.
    cpu : int, optional
        Number of CPU cores, by default 4.
    gpu : int, optional
        Number of GPU cores, by default 1.
    memory : int, optional
        Number of memory in MiB, by default 102400.
    private_machine : str, optional
        Schedule to private machine. Optional: yes, no, group, project, tenant,
        by default "group".
    charged_group : str, optional
        Charged quota group, by default "health".
    max_wait_time : int, optional
        If resource or quota is temporarily unavailable, how long to wait for
        them to become available, by default 99999999.
    preemptible : str, optional
        Whether the worker can be preempted, support in-replica, yes, no,
        best-effort, by default "in-replica"
    log_path : str, optional
        Write outputs of rlaunch into the log file, by default use `sys.stdout`.

    Returns
    -------
    str
        The final command to be executed.
    """
    command = (
        f"rlaunch"
        f" --cpu={cpu}"
        f" --gpu={gpu}"
        f" --memory={memory}"
        f" --private-machine={private_machine}"
        f" --charged-group={charged_group}"
        f" --max-wait-time={max_wait_time}"
        f" --preemptible={preemptible}"
        f" -- {command}"
    )
    if log_path is not None:
        command = f"{command} >> {log_path}"
    return command


def cuda_visible_devices_wrapper(command, device_ids=[]):
    """Use cuda_visible_devices to run a comand.

    Parameters
    ----------
    command : str
        The command to be executed.
    device_ids : list, optional
        The device ids to be used, by default [].

    Returns
    -------
    str
        The final command to be executed.
    """
    if len(device_ids) == 0:
        return command
    device_ids = ",".join(map(str, device_ids))
    command = f"CUDA_VISIBLE_DEVICES={device_ids} {command}"
    return command
