"""One authorized Windows shutdown after this exact training process completes."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/quality_recovery'
AUTH = RUN / 'auto_shutdown_authorization.json'
STATE = RUN / 'auto_shutdown_status.json'
CANCEL = RUN / 'CANCEL_AUTO_SHUTDOWN'
LOCK = RUN / 'auto_shutdown.lock'
SHUTDOWN = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/shutdown.exe'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(state, **values):
    write(STATE, {'state': state, 'utc': datetime.now(timezone.utc).isoformat(),
                  'watcher_pid': os.getpid(), **values})


def api():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    return kernel


def open_process(kernel, pid):
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    creation = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return handle, creation


def completion_evidence():
    result_path = ROOT / 'reports/quality_recovery/03b_training.json'
    result = read(result_path)
    protocol_path = ROOT / 'planning/QUALITY_RECOVERY_CONTINUATION_PROTOCOL.json'
    protocol = read(protocol_path)
    if (result['status'] != 'complete' or result['successful_updates'] != 48000
            or result['stage_successful_updates'] != 32000
            or protocol['overall_updates'] != 48000
            or result['continuation_protocol_sha256'] != digest(protocol_path)
            or result['checkpoint']['completed_updates'] != 32000):
        raise ValueError('The declared complete training outcome is not verified.')
    artifacts = {}
    for name in ['checkpoint', 'inference']:
        artifact = result[name]
        path = Path(artifact['path']).resolve()
        if not path.is_relative_to((ROOT / 'checkpoints').resolve()):
            raise ValueError('Completion artifact is outside this project checkpoint folder.')
        actual = digest(path)
        if actual != artifact['sha256']:
            raise ValueError(f'Final {name} bytes differ from the completed outcome.')
        artifacts[name] = {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}
    tokenizer = ROOT / 'artifacts/quality_recovery/tokenizer.json'
    if digest(tokenizer) != result['identity']['tokenizer_sha256']:
        raise ValueError('Final tokenizer identity differs.')
    return {'successful_updates': result['successful_updates'], 'best_update': result['best_update'],
            'best_development_nll': result['best_development_nll'], 'artifacts': artifacts,
            'completed_report_sha256': digest(result_path),
            'scope': 'training and saved artifacts complete; final evaluation, story review and app update remain pending'}


def arm(pid):
    if AUTH.exists():
        raise ValueError('An authorization already exists. Inspect or cancel it before rearming.')
    kernel = api()
    handle, created = open_process(kernel, pid)
    try:
        if kernel.WaitForSingleObject(handle, 0) != 258:
            raise ValueError('The selected training process is already finished.')
    finally:
        kernel.CloseHandle(handle)
    RUN.mkdir(parents=True, exist_ok=True)
    authorization = {'training_pid': pid, 'training_creation_filetime': created,
                     'script_sha256': digest(Path(__file__)), 'consumed': False,
                     'expected_global_updates': 48000, 'shutdown_countdown_seconds': 120,
                     'maximum_watch_seconds': 21600,
                     'authorization': 'User explicitly requested laptop power off after training completes, 2026-10-22',
                     'scope': 'automatic shutdown after verified successful training; later qualitative review/app work is pending'}
    write(AUTH, authorization)
    record('armed_pending_watcher', training_pid=pid)
    print(json.dumps(authorization, indent=2))


def watch():
    authorization = read(AUTH)
    if authorization['consumed'] or authorization['script_sha256'] != digest(Path(__file__)):
        raise ValueError('Authorization is consumed or helper source changed.')
    RUN.mkdir(parents=True, exist_ok=True)
    lock = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock, str(os.getpid()).encode('ascii'))
    os.close(lock)
    kernel = api()
    handle = None
    try:
        handle, creation = open_process(kernel, authorization['training_pid'])
        if creation != authorization['training_creation_filetime']:
            raise ValueError('The process ID now refers to another process.')
        record('waiting_for_training', training_pid=authorization['training_pid'])
        kernel.SetThreadExecutionState(0x80000001)
        deadline = time.monotonic() + authorization['maximum_watch_seconds']
        while True:
            if CANCEL.exists():
                record('cancelled_before_shutdown')
                return
            if time.monotonic() > deadline:
                raise TimeoutError('Watch allowance expired; no shutdown requested.')
            outcome = kernel.WaitForSingleObject(handle, 30000)
            if outcome == 0:
                break
            if outcome != 258:
                raise ctypes.WinError(ctypes.get_last_error())
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 0:
            raise RuntimeError(f'Training did not exit successfully: {exit_code.value}. No shutdown requested.')
        evidence = completion_evidence()
        if CANCEL.exists():
            record('cancelled_after_training', evidence=evidence)
            return
        authorization['consumed'] = True
        write(AUTH, authorization)
        record('verified_complete_requesting_shutdown', evidence=evidence)
        command = [str(SHUTDOWN), '/s', '/t', '120', '/d', 'p:0:0', '/c',
                   'StorySeed training finished. Final checkpoint and model verified. Automatic shutdown in 2 minutes.']
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f'Windows rejected shutdown: {process.returncode}: {process.stdout} {process.stderr}')
        record('shutdown_scheduled', evidence=evidence, countdown_seconds=120,
               cancel_command='shutdown /a')
        for _ in range(120):
            if CANCEL.exists():
                aborted = subprocess.run([str(SHUTDOWN), '/a'], capture_output=True, text=True)
                record('cancellation_requested_during_countdown', abort_exit_code=aborted.returncode)
                return
            time.sleep(1)
    except Exception as error:
        record('failed_without_shutdown', error=str(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        if handle:
            kernel.CloseHandle(handle)
        LOCK.unlink(missing_ok=True)


def cancel():
    CANCEL.parent.mkdir(parents=True, exist_ok=True)
    CANCEL.touch()
    state = read(STATE) if STATE.exists() else {}
    if state.get('state') == 'shutdown_scheduled':
        subprocess.run([str(SHUTDOWN), '/a'], capture_output=True, text=True)
    print('StorySeed automatic shutdown cancellation requested.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--arm', type=int, metavar='VERIFIED_TRAINING_PID')
    mode.add_argument('--watch', action='store_true')
    mode.add_argument('--cancel', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.arm is not None:
        arm(args.arm)
    elif args.watch:
        watch()
    elif args.cancel:
        cancel()
    else:
        print(json.dumps(read(STATE), indent=2))
