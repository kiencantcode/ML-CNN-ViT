$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "E:\conda-envs\cnn-vit\python.exe"
$log = "outputs\logs\run_tuned.log"
$runs = @("frac1.0_augbasic_seed0", "frac1.0_augbasic_seed1", "frac1.0_augbasic_seed2")

function Step($name, $arguments) {
    "[$(Get-Date -Format s)] $name" | Tee-Object -Append $log
    & $py -u -W ignore @arguments *>> $log
}

Step "experiments" @("scripts\run_experiments.py", "--tuned", "--log", "outputs/logs/run_experiments_tuned.log")
Step "CIFAR-10-C" (@("scripts\eval_cifar10c.py", "--outputs", "outputs/tuned", "--out", "results/tuned/cifar10c", "--runs") + $runs)
Step "Grad-CAM" @("scripts\visualize.py", "--outputs", "outputs/tuned", "--out", "results/tuned/figures/gradcam")
Step "summary" @("scripts\summarize_results.py", "--outputs", "outputs/tuned", "--results", "results/tuned")
"[$(Get-Date -Format s)] all done" | Tee-Object -Append $log
