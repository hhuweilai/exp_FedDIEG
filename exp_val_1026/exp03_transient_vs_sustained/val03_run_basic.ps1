# 验证实验 3（val03）一键跑完（Windows / PowerShell）：短期兴趣扰动 vs 持续兴趣变化
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\val03_run_basic.ps1
#
# 默认行为（一条命令跑完所有结果）：
#   ① 对两个数据集（MovieLens-1M + amazon-digital music）构造场景：
#      真实 arm 的 sustained / transient / stable 队列 + 注入 arm 的 inj / pair 配对
#      （稀疏数据集的真实队列会自动放宽阈值；不可用的会自动跳过，不拖垮整轮）
#   ② 两个场景（sustained / transient）× 3 个 ρ × 3 个随机种子 × 参与率 100%
#   ③ 自动聚合并出图 → results/aggregated/
#
#   已完成的任务自动跳过（中断后重跑同一条命令即可续跑）；单个任务失败不中断整轮。
#
# 成本参考（单次运行 = 1 个数据集 × 1 个场景 × 1 个 ρ × 1 个种子，5 窗口 × 120 轮）：
#   MovieLens-1M 约 3 分钟 → 2×3×3=18 次 ≈ 1 小时
#   amazon-digital music 约 7 分钟 → 18 次 ≈ 2 小时
#   默认两个数据集全跑约 3 小时。想先快速看趋势：
#     -File .\val03_run_basic.ps1 -Rhos "0,0.8" -Seeds "42"      # 约 1 小时
#
# 图强度 / 新边初始化（与验证实验二同一套开关，方便把两个实验的数字并排读）：
#   -GraphStrength paper   已发表配置（默认）
#   -GraphStrength strong  放大图影响（敏感性对照）
#   -NewEdgeInit ema       忠实复刻原实现（默认）
#   -NewEdgeInit warm      新边不被惩罚（会消掉"图冻住"的机制，ρ 之间的权衡才看得出来）
#   两组结果分别写到 results/raw_strong / results/raw_warm，末尾各自聚合。
#
# 注意：本文件须以 UTF-8 with BOM 保存，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错。

param(
    [string] $Datasets   = "movielens-1m,amazon-digital music",
    [string] $Seeds      = "42,777,2024",
    [string] $Rhos       = "0,0.5,0.8",
    [int]    $NumWindows = 5,
    [int]    $Rounds     = 120,
    [ValidateSet("paper", "strong", "all")]
    [string] $GraphStrength = "paper",
    [ValidateSet("ema", "warm", "zero", "all")]
    [string] $NewEdgeInit = "ema",
    [string] $Gate       = "raw",
    [int]    $MinInjectPairs = 8,      # 注入配对数少于此值就跳过该数据集（实验三在它上面不可靠）
    [switch] $NoAggregate              # 只想跑训练、不想聚合
)

$ErrorActionPreference = "Stop"
$Root      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ValDir    = Join-Path $Root "exp_val_1026\exp03_transient_vs_sustained"
$Builder   = Join-Path $ValDir "val03_build_scenarios.py"
$Runner    = Join-Path $ValDir "val03_run_scenario.py"
$Aggregate = Join-Path $ValDir "val03_aggregate.py"
$PlanRoot  = Join-Path $ValDir "results\scenarios"
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
    $parts = @()
    if ($Strength -eq "strong") { $parts += "strong" }
    if ($Init -ne "ema") { $parts += $Init }
    if ($parts.Count -eq 0) { return "" }
    return "_" + ($parts -join "_")
}

$DatasetList  = Split-List $Datasets -KeepSpaces      # 数据集名可能含空格
$SeedList     = Split-List $Seeds | ForEach-Object { [int] $_ }
$RhoList      = Split-List $Rhos  | ForEach-Object { [double]::Parse($_, $Invariant) }
$StrengthList = if ($GraphStrength -eq "all") { @("paper", "strong") } else { @($GraphStrength) }
$InitList     = if ($NewEdgeInit -eq "all") { @("ema", "warm") } else { @($NewEdgeInit) }
$ScenarioList = @("sustained", "transient")

Write-Host "=============================================================="
Write-Host "验证实验 3（val03）一键运行"
Write-Host "  数据集   : $($DatasetList -join ' | ')"
Write-Host "  场景     : $($ScenarioList -join ' / ')"
Write-Host "  rho      : $($RhoList -join ' / ')"
Write-Host "  种子     : $($SeedList -join ' / ')"
Write-Host "  窗口/轮数: $NumWindows 窗口 x $Rounds 轮"
Write-Host "  图强度   : $($StrengthList -join ' + ')"
Write-Host "  新边初值 : $($InitList -join ' + ')"
Write-Host "=============================================================="

# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------
Write-Host "=== 环境自检：$Python ==="
& $Python -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw '环境自检失败：这个 Python 里缺少 numpy / pandas / torch。请先把 $env:PYTHON 指向跑实验用的解释器（例如 D:\Myenvs\FedDIEG\python.exe）。'
}

# ---------------------------------------------------------------------------
# ① 场景构造（计划对所有 ρ / 种子固定）
# ---------------------------------------------------------------------------
$Skipped = @()
foreach ($Dataset in $DatasetList) {
    $slug  = Get-Slug $Dataset
    $plan  = Join-Path $PlanRoot "$slug\val03_injection_plan.csv"
    $real  = Join-Path $PlanRoot "$slug\val03_real_cohorts.csv"
    if ((Test-Path $plan) -and (Test-Path $real)) {
        Write-Host "[skip] 场景已存在：$plan"
    }
    else {
        Write-Host ""
        Write-Host "=== ① 场景构造：$Dataset ==="
        & $Python $Builder --dataset $Dataset --num-windows $NumWindows
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "场景构造失败：$Dataset（跳过这个数据集，继续其它的）"
        }
    }
    if (-not (Test-Path $plan)) {
        Write-Host "[skip] $Dataset 没有生成注入计划，跳过"
        $Skipped += $Dataset
        continue
    }
    $nInj = @(Import-Csv $plan | Where-Object { $_.group -eq "inj" }).Count
    Write-Host "[plan] $Dataset : 注入组 $nInj 人"
    if ($nInj -lt $MinInjectPairs) {
        Write-Host "[skip] $Dataset 的注入组只有 $nInj 人（< $MinInjectPairs），实验三在它上面不可靠，跳过"
        $Skipped += $Dataset
    }
}

$RunDatasets = @($DatasetList | Where-Object { $Skipped -notcontains $_ })
if ($RunDatasets.Count -eq 0) {
    throw "所有数据集的场景都不可用，无法运行实验三。请检查 val03_build_scenarios.py 的输出。"
}

# ---------------------------------------------------------------------------
# ② 训练（数据集 × 场景 × ρ × 种子 × 图强度 × 新边初值）
# ---------------------------------------------------------------------------
$Total   = $RunDatasets.Count * $ScenarioList.Count * $SeedList.Count * $RhoList.Count * $StrengthList.Count * $InitList.Count
$Done    = 0
$Failed  = @()
$StartAt = Get-Date

foreach ($Strength in $StrengthList) {
    foreach ($Init in $InitList) {
        $tag     = Get-CaseTag $Strength $Init
        $OutRoot = Join-Path $ValDir ("results\raw" + $tag)
        $AggOut  = Join-Path $ValDir ("results\aggregated" + $tag)
        Write-Host ""
        Write-Host "---------- 图强度=$Strength 新边初值=$Init → $OutRoot ----------"

        foreach ($Dataset in $RunDatasets) {
            $slug = Get-Slug $Dataset
            foreach ($Scenario in $ScenarioList) {
                foreach ($rho in $RhoList) {
                    $rhoStr = $rho.ToString('0.00', $Invariant)
                    foreach ($Seed in $SeedList) {
                        $Done++
                        $outDir = Join-Path $OutRoot "$slug\$Scenario\rho$rhoStr\seed_$Seed"
                        if (Test-Path (Join-Path $outDir "val03_summary.csv")) {
                            Write-Host "[$Done/$Total][skip] $Dataset | $Scenario | rho=$rhoStr | seed=$Seed 已完成"
                            continue
                        }
                        Write-Host ""
                        Write-Host "[$Done/$Total] $Dataset | $Scenario | rho=$rhoStr | seed=$Seed | $Strength/$Init"
                        & $Python $Runner `
                            --dataset $Dataset `
                            --scenario $Scenario `
                            --rho $rhoStr `
                            --seed $Seed `
                            --num-windows $NumWindows `
                            --rounds-per-window $Rounds `
                            --participation-rate 1.0 `
                            --graph-strength $Strength `
                            --graph-gate $Gate `
                            --new-edge-init $Init `
                            --out-root $OutRoot
                        if ($LASTEXITCODE -ne 0) {
                            Write-Warning "运行失败：$Dataset | $Scenario | rho=$rhoStr | seed=$Seed | $Strength/$Init（继续）"
                            $Failed += "$Dataset|$Scenario|rho=$rhoStr|seed=$Seed|$Strength|$Init"
                        }
                    }
                }
            }
        }

        # -------------------------------------------------------------------
        # ③ 聚合
        # -------------------------------------------------------------------
        if (-not $NoAggregate) {
            Write-Host ""
            Write-Host "=== ③ 聚合：$OutRoot → $AggOut ==="
            & $Python $Aggregate --root $OutRoot --out $AggOut
            if ($LASTEXITCODE -ne 0) { Write-Warning "聚合失败：$OutRoot" }
        }
    }
}

$Elapsed = (Get-Date) - $StartAt
Write-Host ""
Write-Host "=============================================================="
Write-Host ("全部完成，用时 {0:hh\:mm\:ss}" -f $Elapsed)
if ($Failed.Count -gt 0) {
    Write-Host "失败的组合（重跑同一条命令会自动续跑）："
    $Failed | ForEach-Object { Write-Host "  - $_" }
}
else {
    Write-Host "没有任何失败组合。"
}
Write-Host "结果：$ValDir\results\aggregated\"
Write-Host "=============================================================="
