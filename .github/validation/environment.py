#!/usr/bin/env python3
"""Capture guest hardware, OS and toolchain provenance; never load a model."""
import argparse
import datetime
import json
import os
import platform
import struct
import subprocess
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('platform', choices=['linux', 'windows'])
ap.add_argument('output', type=Path)
a = ap.parse_args()


def probe(argv, required=False):
    try:
        r = subprocess.run(argv, input='', text=True, encoding='utf-8', errors='replace',
                           capture_output=True, timeout=45)
        result = {'command': argv, 'exit_code': r.returncode,
                  'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {'command': argv, 'error': str(exc), 'exit_code': None}
    if required and result['exit_code'] != 0:
        raise RuntimeError('Required environment probe failed: ' + repr(result))
    return result


def text_file(name):
    try:
        return Path(name).read_text(encoding='utf-8').strip()
    except OSError:
        return None


info = {
    'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform': a.platform,
    'python_platform': platform.platform(),
    'architecture': platform.machine(),
    'pointer_bits': struct.calcsize('P') * 8,
    'logical_cpus_visible_to_python': os.cpu_count(),
    'guest_hardware_only': True,
    'note': 'Hosted VM observations, not a claim about the underlying physical CPU allocation or storage device.',
    'workspace': str(Path.cwd()),
    'temporary_directory': tempfile.gettempdir(),
    'runner': {k: os.environ.get(k) for k in
               ['RUNNER_OS', 'RUNNER_ARCH', 'ImageOS', 'ImageVersion', 'GITHUB_REPOSITORY',
                'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REF_NAME']},
    'correctness_commit': 'e2a0868ef57630883f1269e765ee4a69a366409d',
    'base_commit': '1efb7c82a0c473984cc0258bfc9e12c0cae94914',
    'test_configuration': {
        'model_inference': False, 'real_gpu_execution': False, 'performance_benchmark': False,
        'compiled_isa': 'x86-64-v3', 'openmp_threads': 2, 'repetitions_per_mode': 3,
        'modes': ['OpenMP codegen disabled (-fno-openmp), runtime still linked', 'OpenMP enabled'],
        'fixture': {'hidden': 128, 'intermediate': 64, 'experts': 32, 'top_k': 8, 'group_size': 64,
                    'weights': 'synthetic planar offset-encoded INT4', 'scales': 'FP32',
                    'activations': 'FP32', 'checkpoint': None},
        'cache_capacities': [1, 2, 7, 8, 15, 16],
        'linux_sanitizers': 'Clang ASan/UBSan and TSan without OpenMP; leak checking off; TSan process ASLR off',
    },
    'tools': {
        'gcc': probe(['gcc', '--version'], True),
        'gcc_configuration': probe(['gcc', '-v'], True),
        'make': probe(['make', '--version'], True),
        'git': probe(['git', '--version'], True),
        'python': platform.python_version(),
    },
}
macros = probe(['gcc', '-march=x86-64-v3', '-fopenmp', '-dM', '-E', '-x', 'c', '-'], True)
wanted = {'__AVX__', '__AVX2__', '__FMA__', '__SSE4_2__', '__x86_64__', '_OPENMP'}
info['compiler_feature_macros'] = [line for line in macros['stdout'].splitlines()
                                    if len(line.split()) >= 2 and line.split()[1] in wanted]
if a.platform == 'linux':
    info['os_release'] = text_file('/etc/os-release')
    info['kernel'] = platform.uname()._asdict()
    cpu = probe(['lscpu', '--json'], True)
    info['cpu'] = json.loads(cpu['stdout'])
    memory = {}
    for line in (text_file('/proc/meminfo') or '').splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key + '_bytes'] = int(value.split()[0]) * 1024
    if 'MemTotal_bytes' not in memory:
        raise RuntimeError('Cannot read Linux guest RAM')
    info['memory'] = memory
    info['cgroup_limits'] = {name: text_file('/sys/fs/cgroup/' + name)
                             for name in ['memory.max', 'cpu.max']}
    info['virtual_machine'] = {name: text_file('/sys/class/dmi/id/' + name)
                               for name in ['sys_vendor', 'product_name', 'product_version']}
    info['filesystems'] = probe(['df', '-T', '.', tempfile.gettempdir()], True)
    info['tools']['clang'] = probe(['clang', '--version'], True)
    info['tools']['glibc'] = probe(['ldd', '--version'], True)
else:
    ps = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@{
  os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory
  computer = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory,NumberOfLogicalProcessors,HypervisorPresent
  processors = @(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,AddressWidth,MaxClockSpeed)
  disks = @(Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,DriveType,FileSystem,Size,FreeSpace)
} | ConvertTo-Json -Depth 6 -Compress
'''
    native = probe(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', ps], True)
    info['windows_cim'] = json.loads(native['stdout'])
    info['msys'] = probe(['uname', '-a'], True)
    info['msys_environment'] = {k: os.environ.get(k) for k in ['MSYSTEM', 'MINGW_PREFIX', 'MSYS2_PATH_TYPE']}
    # Optional .NET guest ISA probe, distinct from compiler target macros.
    isa = probe(['pwsh.exe', '-NoProfile', '-NonInteractive', '-Command',
                 '@{AVX2=[System.Runtime.Intrinsics.X86.Avx2]::IsSupported; FMA=[System.Runtime.Intrinsics.X86.Fma]::IsSupported} | ConvertTo-Json -Compress'])
    info['guest_isa_probe'] = isa

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_bytes((json.dumps(info, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
print('Environment provenance written to', a.output)
