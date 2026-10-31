# 验证实验 1（val01）批次脚本（Windows / PowerShell）
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val01_run_basic.ps1
#   powershell ... -File .\val01_run_basic.ps1 -Datasets "movielens-1m"      # 只跑一个数据集
#   powershell ... -File .\val01_run_basic.ps1 -Seeds "42" -Rounds 60        # 先探路
#
# 网格（符合任务书"实验一"要求）：
#   upload_only        p = 1.0 / 0.5 / 0.3 / 0.1
#   upload_fresh_edges p = 0.3 / 0.1        ← 对照：只把"信息是否新鲜"换掉
#   数据集：MovieLens-1M + 一个 Amazon 数据集；3 个种子；6 组 × 3 × 2 = 36 次运行
#   每次运行只写 4 个小文件（约 10 KB）
#
# 注意：本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

param(
    [string] $Datasets   = "movielens-1m,amazon-digital music",
    [string] $Seeds      = "42,777,2024",
    [int]    $NumWindows = 5,
    [int]    $Rounds     = 120,
    [string] $FullRates  = "1.0,0.5,0.3,0.1",
    [string] $CtrlRates  = "0.3,0.1"
)

$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Script = Join-Path $Root "exp_val_1026\exp01_staleness_participation\val01_run_staleness.py"
$Out    = Join-Path $Root "exp_val_1026\exp01_staleness_participation\results\raw"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Split-List([string] $Value, [switch] $KeepSpaces) {
    $pattern = if ($KeepSpaces) { ',' } else { '[,;\s]+' }
    return @($Value -split $pattern | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

$DatasetList  = Split-List $Datasets -KeepSpaces      # 数据集名可能含空格
$SeedList     = Split-List $Seeds      | ForEach-Object { [int] $_ }
$FullRateList = Split-List $FullRates  | ForEach-Object { [double] $_ }
$CtrlRateList = Split-List $CtrlRates  | ForEach-Object { [double] $_ }

function Invoke-Run([string] $Dataset, [string] $Mode, [double] $Rate, [int] $Seed) {
    $slug    = ($Dataset -replace '[^A-Za-z0-9_\-]', '_').Trim('_')
    $rateStr = $Rate.ToString("0.00")
    $done    = Join-Path $Out "$slug\$Mode\p$rateStr\seed_$Seed\val01_summary.csv"
    if (Test-Path $done) { Write-Host "[skip] $done 已存在"; return }

    Write-Host "=== $Dataset | $Mode | p=$rateStr | seed=$Seed ==="
    & $Python $Script `
        --dataset $Dataset `
        --mode $Mode `
        --participation-rate $Rate `
        --seed $Seed `
        --num-windows $NumWindows `
        --rounds-per-window $Rounds `
        --out-root $Out
    if ($LASTEXITCODE -ne 0) { throw "run failed: $Dataset $Mode p$rateStr seed$Seed" }
}

foreach ($Dataset in $DatasetList) {
    foreach ($Seed in $SeedList) {
        foreach ($Rate in $FullRateList) { Invoke-Run $Dataset "upload_only" $Rate $Seed }
        foreach ($Rate in $CtrlRateList) { Invoke-Run $Dataset "upload_fresh_edges" $Rate $Seed }
    }
}

$Agg = Join-Path $Root "exp_val_1026\exp01_staleness_participation\val01_aggregate.py"
Write-Host ""
Write-Host "跑完。出图 + 结论："
Write-Host "  $Python $Agg"
