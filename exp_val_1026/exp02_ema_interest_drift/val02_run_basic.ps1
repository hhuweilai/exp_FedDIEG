# 验证实验 2（val02）一键跑完（Windows / PowerShell）：兴趣真实变化时，固定 EMA 是否适应太慢
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val02_run_basic.ps1
#
# 默认行为（一条命令跑完所有结果）：
#   ① 对两个数据集（MovieLens-1M + amazon-digital music）做离线漂移识别
#      （稀疏数据集的 cohort 会自动放宽阈值；实在不可用的数据集自动跳过，不拖垮整轮）
#   ② 4 个 ρ（0 / 0.5 / 0.8 / 0.95）× 3 个随机种子（42/777/2024）× 参与率 100%
#   ③ 自动聚合并出图 → results/aggregated/
#
#   已完成的任务会自动跳过，中断后重跑同一条命令即可续跑；单个任务失败不会中断整轮。
#
# 图强度：
#   -GraphStrength paper   只跑已发表配置（默认，约 2 小时）
#   -GraphStrength strong  只跑放大图影响的敏感性配置
#   -GraphStrength all     两组都跑（约 4 小时），分别出到 results/aggregated 与 results/aggregated_strong
#
# 注意：本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

param(
    [string] $Datasets   = "movielens-1m,amazon-digital music",
    [string] $Seeds      = "42,777,2024",
    [string] $Rhos       = "0,0.5,0.8,0.95",
    [int]    $NumWindows = 5,
    [int]    $Rounds     = 120,
    [ValidateSet("paper", "strong", "mix", "all")]
    [string] $GraphStrength = "paper",
    [string] $Gate       = "raw",
    [string] $NewEdgeInit = "ema",
    [int]    $MinDriftCohort = 10,     # drift cohort 少于此人数就跳过该数据集（实验二在它上面不可靠）
    [switch] $NoAggregate              # 只想跑训练、不想聚合时加这个开关
)

$ErrorActionPreference = "Stop"
$Root      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ValDir    = Join-Path $Root "exp_val_1026\exp02_ema_interest_drift"
$Detect    = Join-Path $ValDir "val02_detect_drift.py"
$Runner    = Join-Path $ValDir "val02_run_ema.py"
$Aggregate = Join-Path $ValDir "val02_aggregate.py"
$RelationQuality = Join-Path $ValDir "val02_relation_quality.py"
$DriftRoot = Join-Path $ValDir "results\drift"
$Python    = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Split-List([string] $Value, [switch] $KeepSpaces) {
    $pattern = if ($KeepSpaces) { ',' } else { '[,;\s]+' }
    return @($Value -split $pattern | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

function Get-Slug([string] $Name) {
    return ($Name -replace '[^A-Za-z0-9_\-]', '_').Trim('_')
}

function Get-CaseTag([string] $Strength, [string] $Init) {
    # 目录后缀：paper+ema 用默认目录（与已跑结果兼容），其余组合各自独立，避免互相覆盖/被误判为已完成
    $parts = @()
    if ($Strength -ne "paper") { $parts += $Strength }
    if ($Init -ne "ema") { $parts += $Init }
    if ($parts.Count -eq 0) { return "" }
    return "_" + ($parts -join "_")
}

function Get-OutRoot([string] $Strength, [string] $Init) {
    return (Join-Path $ValDir ("results\raw" + (Get-CaseTag $Strength $Init)))
}

function Get-AggOut([string] $Strength, [string] $Init) {
    return (Join-Path $ValDir ("results\aggregated" + (Get-CaseTag $Strength $Init)))
}

$DatasetList = Split-List $Datasets -KeepSpaces      # 数据集名可能含空格
$SeedList    = Split-List $Seeds | ForEach-Object { [int] $_ }
$RhoList     = Split-List $Rhos  | ForEach-Object { [double]::Parse($_, $Invariant) }
$StrengthList = if ($GraphStrength -eq "all") { @("paper", "strong", "mix") } else { @($GraphStrength) }

Write-Host "=============================================================="
Write-Host "验证实验 2（val02）一键运行"
Write-Host "  数据集   : $($DatasetList -join ' | ')"
Write-Host "  rho      : $($RhoList -join ' / ')"
Write-Host "  种子     : $($SeedList -join ' / ')"
Write-Host "  窗口/轮数: $NumWindows 窗口 x $Rounds 轮"
Write-Host "  图强度   : $($StrengthList -join ' + ')"
Write-Host "=============================================================="

# ---------------------------------------------------------------------------
# 环境自检
# "失败继续"的代价是：环境不对时脚本会一路跑到底、最后才看到一堆失败。
# 所以开跑前先确认这个解释器里真的有 numpy / pandas / torch，缺了就直接停。
# ---------------------------------------------------------------------------
Write-Host "=== 环境自检：$Python ==="
& $Python -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw '环境自检失败：这个 Python 里缺少 numpy / pandas / torch。请先把 $env:PYTHON 指向跑实验用的解释器（例如 D:\Myenvs\FedDIEG\python.exe）。'
}

# ---------------------------------------------------------------------------
# ① 离线漂移识别（cohort 对所有 rho / 种子固定；稀疏数据集自动放宽阈值）
# ---------------------------------------------------------------------------
$Skipped = @()
foreach ($Dataset in $DatasetList) {
    $slug   = Get-Slug $Dataset
    $cohort = Join-Path $DriftRoot "$slug\val02_drift_users.csv"
    if (Test-Path $cohort) {
        Write-Host "[skip] cohort 已存在：$cohort"
    }
    else {
        Write-Host ""
        Write-Host "=== ① 漂移识别：$Dataset ==="
        & $Python $Detect --dataset $Dataset --num-windows $NumWindows
        if ($LASTEXITCODE -ne 0) {
            # 单个数据集的识别失败不终止整轮：下面 Test-Path 判定失败后会自动跳过它
            Write-Warning "漂移识别失败：$Dataset（跳过这个数据集，继续其它的）"
        }
    }
    if (Test-Path $cohort) {
        $nDrift = @(Import-Csv $cohort | Where-Object { $_.cohort -eq "drift" }).Count
        Write-Host "[cohort] $Dataset : drift = $nDrift 人"
        if ($nDrift -lt $MinDriftCohort) {
            Write-Host "[skip] $Dataset 的 drift cohort 只有 $nDrift 人（< $MinDriftCohort），该数据集上实验二不可靠，跳过"
            $Skipped += $Dataset
        }
    }
    else {
        Write-Host "[skip] $Dataset 没有生成 cohort，跳过"
        $Skipped += $Dataset
    }
}

$RunDatasets = @($DatasetList | Where-Object { $Skipped -notcontains $_ })
if ($RunDatasets.Count -eq 0) {
    throw "所有数据集的 cohort 都不可用，无法运行实验二。请检查 val02_detect_drift.py 的输出。"
}

# ---------------------------------------------------------------------------
# ② 训练（数据集 × rho × 种子 × 图强度）
# ---------------------------------------------------------------------------
$Total = $RunDatasets.Count * $SeedList.Count * $RhoList.Count * $StrengthList.Count
$Done = 0
$Failed = @()
$StartTime = Get-Date

foreach ($Strength in $StrengthList) {
    $OutRoot = Get-OutRoot $Strength $NewEdgeInit
    foreach ($Dataset in $RunDatasets) {
        $slug = Get-Slug $Dataset
        foreach ($Seed in $SeedList) {
            foreach ($Rho in $RhoList) {
                $Done += 1
                $rhoStr = $Rho.ToString('0.00', $Invariant)
                $doneFile = Join-Path $OutRoot "$slug\rho$rhoStr\seed_$Seed\val02_summary.csv"
                if (Test-Path $doneFile) {
                    Write-Host "[$Done/$Total][skip] $Dataset | $Strength | rho=$rhoStr | seed=$Seed 已完成"
                    continue
                }
                Write-Host ""
                Write-Host "[$Done/$Total] === $Dataset | $Strength | rho=$rhoStr | seed=$Seed ==="
                & $Python $Runner `
                    --dataset $Dataset `
                    --rho $rhoStr `
                    --seed $Seed `
                    --num-windows $NumWindows `
                    --rounds-per-window $Rounds `
                    --participation-rate 1.0 `
                    --graph-strength $Strength `
                    --graph-gate $Gate `
                    --new-edge-init $NewEdgeInit `
                    --out-root $OutRoot
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "运行失败：$Dataset | $Strength | rho=$rhoStr | seed=$Seed（继续后面的任务）"
                    $Failed += "$Dataset|$Strength|rho$rhoStr|seed$Seed"
                }
            }
        }
    }
}

# ---------------------------------------------------------------------------
# ③ 自动聚合出图
# ---------------------------------------------------------------------------
$Elapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "=============================================================="
Write-Host ("训练阶段结束，用时 {0:hh\:mm\:ss}" -f $Elapsed)
if ($Skipped.Count -gt 0) { Write-Host ("跳过的数据集: " + ($Skipped -join ", ")) }
if ($Failed.Count -gt 0) { Write-Host ("失败的运行: " + ($Failed -join "; ")) }

if (-not $NoAggregate) {
    foreach ($Strength in $StrengthList) {
        $OutRoot = Get-OutRoot $Strength $NewEdgeInit
        $AggOut = Get-AggOut $Strength $NewEdgeInit
        $hasRun = Get-ChildItem -Path $OutRoot -Recurse -Filter val02_summary.csv -ErrorAction SilentlyContinue
        if (-not $hasRun) {
            Write-Host "[skip] $Strength / $NewEdgeInit 没有可用结果，跳过聚合"
            continue
        }
        Write-Host ""
        Write-Host "=== ③ 聚合（$Strength / new-edge-init=$NewEdgeInit）-> $AggOut ==="
        & $Python $Aggregate --root $OutRoot --out $AggOut
        if ($LASTEXITCODE -ne 0) { Write-Warning "聚合失败：$Strength / $NewEdgeInit" }

        Write-Host "--- 关系质量分析（邻居兴趣一致度 vs 随机基线）---"
        & $Python $RelationQuality --root $OutRoot --out $AggOut
        if ($LASTEXITCODE -ne 0) { Write-Warning "关系质量分析失败：$Strength / $NewEdgeInit" }
    }
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "结果位置（本组：$GraphStrength / new-edge-init=$NewEdgeInit）："
$FinalAgg = Get-AggOut $StrengthList[-1] $NewEdgeInit
$FinalOut = Get-OutRoot $StrengthList[-1] $NewEdgeInit
Write-Host ("  " + (Join-Path $FinalAgg "figures\val02_fig_ema_recovery.png") + "   <- 任务书要的那张恢复曲线")
Write-Host ("  " + (Join-Path $FinalAgg "figures\val02_fig_edge_adjust.png") + "    <- 旧兴趣同伴占比 / 邻居 turnover")
Write-Host ("  " + (Join-Path $FinalAgg "tables\val02_lag_table.csv") + "           <- 各 rho 的滞后指标")
Write-Host ("  " + (Join-Path $FinalAgg "tables\val02_verdict.csv") + "             <- 自动结论判定")
Write-Host ("  " + (Join-Path $FinalAgg "val02_report.md") + "                      <- 报告骨架")
Write-Host ""
Write-Host "可选的后续对照（各自独立目录，不会覆盖本次结果，已完成的会跳过）："
Write-Host "  -GraphStrength strong       放大的图影响（检验'图权重太小'，ML 约 35 分钟/数据集）"
Write-Host "  -NewEdgeInit warm           新边不被惩罚（检验'冻结'是否来自初始化，ML 约 35 分钟/数据集）"
Write-Host "  -Datasets `"movielens-1m`"    只跑一个数据集"
Write-Host "=============================================================="
