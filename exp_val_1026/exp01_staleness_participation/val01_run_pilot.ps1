# 验证实验 1（val01）· Windows 冒烟脚本（PowerShell）
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val01_run_pilot.ps1
#
# 只用于确认环境能跑通（1 窗口 20 轮、2000 个评测用户），结果无统计意义。
# 本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

$ErrorActionPreference = "Stop"

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Base   = Join-Path $Root "exp_val_1026\exp01_staleness_participation"
$Script = Join-Path $Base "val01_run_staleness.py"
$Out    = Join-Path $Base "results\archive\pilot"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

foreach ($Mode in @("upload_only", "upload_fresh_edges")) {
    Write-Host "=== pilot | $Mode ==="
    & $Python $Script `
        --dataset movielens-1m `
        --mode $Mode `
        --participation-rate 0.3 `
        --seed 42 `
        --num-windows 1 `
        --rounds-per-window 20 `
        --eval-every 10 `
        --max-eval-users 2000 `
        --out-root $Out
    if ($LASTEXITCODE -ne 0) { throw "run failed: $Mode" }
}

Write-Host ""
Write-Host "冒烟完成。检查产物："
Write-Host "  $Out\movielens-1m\<mode>\p0.30\seed_42\"
Write-Host "  应包含 config.json / val01_summary.csv / val01_staleness_buckets.csv / val01_run.log"
