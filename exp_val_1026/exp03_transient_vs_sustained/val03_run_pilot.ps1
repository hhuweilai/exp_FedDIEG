# 验证实验 3（val03）冒烟/探路（Windows / PowerShell）
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val03_run_pilot.ps1
#
# 两个数据集 × 2 个 ρ（0 和 0.8，最容易看出差异的两端）× 1 个种子 × 40 轮/窗口，
# 结果写到 results/archive/pilot_raw（不污染正式结果），末尾自动聚合到
# results/archive/pilot_aggregated。约 15 分钟。
# 目的：正式开跑前先确认「场景构造 + 注入 + 训练 + 聚合出图」整条链路在两个数据集上都通。
#
# 注意：本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

param(
    [string] $Datasets   = "movielens-1m,amazon-digital music",
    [int]    $Seed       = 42,
    [int]    $NumWindows = 5,
    [int]    $Rounds     = 40,
    [string] $Rhos       = "0,0.8"
)

$ErrorActionPreference = "Stop"
$Root      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ValDir    = Join-Path $Root "exp_val_1026\exp03_transient_vs_sustained"
$Builder   = Join-Path $ValDir "val03_build_scenarios.py"
$Runner    = Join-Path $ValDir "val03_run_scenario.py"
$Aggregate = Join-Path $ValDir "val03_aggregate.py"
$Archive   = Join-Path $ValDir "results\archive"
$PilotRoot = Join-Path $Archive "pilot_raw"
$AggOut    = Join-Path $Archive "pilot_aggregated"
$Python    = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

$DatasetList = @($Datasets -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$RhoList = @(
    $Rhos -split '[,;]+' |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne "" } |
        ForEach-Object { [double]::Parse($_, $Invariant) }
)
$ScenarioList = @("sustained", "transient")

# ① 环境自检
Write-Host "=== 环境自检：$Python ==="
& $Python -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw '环境自检失败：这个 Python 里缺少 numpy / pandas / torch。请先把 $env:PYTHON 指向跑实验用的解释器。'
}

# ② 场景构造（写正式目录；正式跑会复用）
foreach ($Dataset in $DatasetList) {
    $slug = ($Dataset -replace '[^A-Za-z0-9_\-]', '_').Trim('_')
    $plan = Join-Path $ValDir "results\scenarios\$slug\val03_injection_plan.csv"
    if (Test-Path $plan) {
        Write-Host "[skip] 场景已存在：$plan"
        continue
    }
    Write-Host "=== 场景构造：$Dataset ==="
    & $Python $Builder --dataset $Dataset --num-windows $NumWindows
    if ($LASTEXITCODE -ne 0) { Write-Warning "场景构造失败：$Dataset（继续）" }
}

# ③ 试跑
foreach ($Dataset in $DatasetList) {
    $slug = ($Dataset -replace '[^A-Za-z0-9_\-]', '_').Trim('_')
    foreach ($Scenario in $ScenarioList) {
        foreach ($rho in $RhoList) {
            $rhoStr = $rho.ToString('0.00', $Invariant)
            $outDir = Join-Path $PilotRoot "$slug\$Scenario\rho$rhoStr\seed_$Seed"
            if (Test-Path (Join-Path $outDir "val03_summary.csv")) {
                Write-Host "[skip] $outDir 已存在"
                continue
            }
            Write-Host "=== pilot | $Dataset | $Scenario | rho=$rhoStr | seed=$Seed | $Rounds 轮/窗口 ==="
            & $Python $Runner `
                --dataset $Dataset `
                --scenario $Scenario `
                --rho $rhoStr `
                --seed $Seed `
                --num-windows $NumWindows `
                --rounds-per-window $Rounds `
                --participation-rate 1.0 `
                --out-dir $outDir
            if ($LASTEXITCODE -ne 0) { Write-Warning "pilot 运行失败：$Dataset $Scenario rho$rhoStr（继续）" }
        }
    }
}

# ④ 试跑结果也聚合一遍，确认出图链路正常
Write-Host ""
Write-Host "=== 聚合 pilot 结果 ==="
& $Python $Aggregate --root $PilotRoot --out $AggOut --rounds-per-window $Rounds
if ($LASTEXITCODE -ne 0) { Write-Warning "pilot 聚合失败" }

Write-Host ""
Write-Host "试跑结束。正式跑用："
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\val03_run_basic.ps1"
