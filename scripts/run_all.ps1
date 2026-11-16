$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "E:\conda-envs\cnn-vit\python.exe"
$log = "outputs\logs\run_all.log"
New-Item -ItemType Directory -Force "outputs\logs" | Out-Null

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_experiments.py*" }
if ($running) {
    Write-Host "run_experiments.py is already running (PID $($running.ProcessId -join ', ')). Not starting a second copy."
    exit 1
}

function Step($name, $arguments) {
    "[$(Get-Date -Format s)] $name" | Tee-Object -Append $log
    & $py -u -W ignore @arguments *>> $log
}

Step "experiments" @("scripts\run_experiments.py")
Step "CIFAR-10-C" @("scripts\eval_cifar10c.py", "--runs", "frac1.0_augbasic_seed0", "frac1.0_augbasic_seed1", "frac1.0_augbasic_seed2")
Step "Grad-CAM" @("scripts\visualize.py")
Step "summary" @("scripts\summarize_results.py")
"[$(Get-Date -Format s)] all done" | Tee-Object -Append $log
