# 验证实验 1 · 三项前置检验（Windows / PowerShell）
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val01_run_checks.ps1
#   powershell ... -File .\val01_run_checks.ps1 -Rounds 60 -Seeds "42"      # 先探路
#
# 三项检验都只跑 MovieLens-1M 的低参与率（0.1 / 0.3）、3 个种子，结果写到
# results\checks\<检验名>\ 下，不会覆盖正式结果 results\raw\。
#
#   检验 1  graphmix0.2    graph_mix 0.05 → 0.2，重跑 upload_only / upload_fresh_edges
#                          判据：陈旧损失若从 ~0.0013 放大到 ≥0.005 → 效应被小图权重淹没
#   检验 2  edgescope_all  每轮重算"全部活跃用户"的邻居（--update-scope all）
#                          判据：若比现状高 ≥0.005 → 真正的问题是"边更新滞后"
#   检验 3  drift          每窗口末量"缓存摘要 vs 新鲜摘要"的余弦/L2（--drift-sample-users）
#                          判据：若 9 轮后 cos 仍在 0.9 以上 → 陈旧程度本就很小
#
# 跑完执行汇总：
#   python val01_checks_report.py
#
# 注意：本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

param(
    [string] $Dataset    = "movielens-1m",
    [string] $Seeds      = "42,777,2024",
    [int]    $NumWindows = 5,
    [int]    $Rounds     = 120,
    [string] $Rates      = "0.1,0.3",
    [int]    $DriftUsers = 2000,
    [string] $Only       = ""
)

$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Base   = Join-Path $Root "exp_val_1026\exp01_staleness_participation"
$Script = Join-Path $Base "val01_run_staleness.py"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Split-List([string] $Value, [switch] $KeepSpaces) {
    $pattern = if ($KeepSpaces) { ',' } else { '[,;\s]+' }
    return @($Value -split $pattern | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

$SeedList = Split-List $Seeds     | ForEach-Object { [int] $_ }
$RateList = Split-List $Rates     | ForEach-Object { [double] $_ }
$Slug     = ($Dataset -replace '[^A-Za-z0-9_\-]', '_').Trim('_')

function Invoke-Check([string] $Check, [string] $Mode, [double] $Rate, [int] $Seed, [string[]] $Extra) {
    $out     = Join-Path $Base "results\checks\$Check"
    $rateStr = $Rate.ToString("0.00")
    $done    = Join-Path $out "$Slug\$Mode\p$rateStr\seed_$Seed\val01_summary.csv"
    if (Test-Path $done) { Write-Host "[skip] $done 已存在"; return }

    Write-Host "=== $Check | $Mode | p=$rateStr | seed=$Seed ==="
    $callArgs = @(
        $Script,
        "--dataset", $Dataset,
        "--mode", $Mode,
        "--participation-rate", $Rate,
        "--seed", $Seed,
        "--num-windows", $NumWindows,
        "--rounds-per-window", $Rounds,
        "--out-root", $out
    ) + $Extra
    & $Python @callArgs
    if ($LASTEXITCODE -ne 0) { throw "run failed: $Check $Mode p$rateStr seed$Seed" }
}

foreach ($Rate in $RateList) {
    foreach ($Seed in $SeedList) {
        # 检验 1：图权重敏感性（两个模式都要跑，才能算 stale vs fresh 的配对差）
        if ($Only -eq "" -or $Only -eq "graphmix") {
            foreach ($Mode in @("upload_only", "upload_fresh_edges")) {
                Invoke-Check "graphmix0.2" $Mode $Rate $Seed @("--feddieg-graph-mix", "0.2")
            }
        }
        # 检验 2：边更新范围（只改"谁的边被重算"，摘要仍可能是旧的）
        if ($Only -eq "" -or $Only -eq "edgescope") {
            Invoke-Check "edgescope_all" "upload_only" $Rate $Seed @("--update-scope", "all")
        }
        # 检验 3：摘要漂移测量
        if ($Only -eq "" -or $Only -eq "drift") {
            Invoke-Check "drift" "upload_only" $Rate $Seed @("--drift-sample-users", "$DriftUsers")
        }
    }
}

$Rep = Join-Path $Base "val01_checks_report.py"
Write-Host ""
Write-Host "三项检验跑完。汇总："
Write-Host "  $Python $Rep"
