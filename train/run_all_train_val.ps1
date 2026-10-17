<#
.SYNOPSIS
Trains YOLO11n and FSDH-YOLO on Dataset B v4 aug for seeds 0/1/2, then
runs independent test, quality, Common Evaluator, latency, and comparison
report generation.

.DESCRIPTION
Formal runs are locked to the audited Dataset B v4 augmented dataset, CIoU,
the established Dataset B native initialization protocol, and seeds 0/1/2.
All train/validation settings remain aligned with the previous Dataset B
comparison: 300 epochs, batch 12, patience 100, and 640-pixel evaluation.

Use -PlanOnly to audit every command without starting work. Debug mode is
isolated from formal outputs, uses one epoch and 2% of the training split, and
deletes only its exact campaign directory after every requested stage succeeds.
#>

param(
    [int]$Epochs = 300,
    [int]$Batch = 12,
    [ValidateRange(0, 8)][int]$Workers = 6,
    [ValidateRange(0, 4)][int]$ValWorkers = 2,
    [int]$ValBatch = 16,
    [int]$Patience = 100,
    [int[]]$Seeds = @(0, 1, 2),
    [ValidateSet("YOLO11n", "FSDH_YOLO")]
    [string[]]$Experiments = @("YOLO11n", "FSDH_YOLO"),
    [switch]$QuietProgress,
    [switch]$SkipLatency,
    [switch]$PlanOnly,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:YOLO_VERBOSE = if ($QuietProgress) { "False" } else { "True" }
$null = & chcp.com 65001

$RepoRoot = $PSScriptRoot
$PyExe = "<python>"
$TrainScript = Join-Path $RepoRoot "origin_train.py"
$ValScript = Join-Path $RepoRoot "val.py"
$QualityScript = Join-Path $RepoRoot "scripts\eval_detection_quality_metrics.py"
$CommonEvalScript = Join-Path $RepoRoot "scripts\eval_ultralytics_common_evaluator.py"
$LatencyScript = Join-Path $RepoRoot "tools\benchmark_uniform_latency.py"
$SummaryScript = Join-Path $RepoRoot "scripts\summarize_dataset_b_common_evaluator.py"
$DatasetB = "<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_aug\lichi_maturity_groupaware_clean_v4_aug.yaml"
$DatasetBTestImages = "<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_aug\test\images"
$DatasetBCommonAnnotations = Join-Path $RepoRoot "external_configs\dataset_b_native4_v4\instances_test.json"
$Weights = Join-Path $RepoRoot "yolo11n.pt"

$AllSpecs = @(
    [pscustomobject]@{
        Id = "YOLO11n"; Label = "YOLO11n"; CommonId = "DatasetB4_YOLO11n"
        Model = $Weights
        RunBase = "dsB_groupaware_clean_v4_aug__comparison__yolo11n"
        TrainInitArgs = @()
    },
    [pscustomobject]@{
        Id = "FSDH_YOLO"; Label = "FSDH-YOLO-n"; CommonId = "DatasetB4_FSDH_YOLO"
        Model = Join-Path $RepoRoot "ultralytics\cfg\models\11\yolo11n_FSDConv-SHAB.yaml"
        RunBase = "dsB_groupaware_clean_v4_aug__comparison__fsdh_yolo_n"
        # Preserve the audited Dataset B protocol: native same-name/same-shape transfer.
        TrainInitArgs = @("--weights", $Weights)
    }
)
$Specs = @($AllSpecs | Where-Object { $_.Id -in $Experiments })
$Campaign = "paper_dataset_b_groupaware_clean_v4_aug_comparison_v1"
$DebugCampaignRoot = Join-Path $RepoRoot "runs\debug\$Campaign"

if ($Debug) {
    $TrainProject = Join-Path $DebugCampaignRoot "train"
    $TestProject = Join-Path $DebugCampaignRoot "test"
    $QualityProject = Join-Path $DebugCampaignRoot "quality"
    $CommonProject = Join-Path $DebugCampaignRoot "common_eval"
    $LatencyProject = Join-Path $DebugCampaignRoot "latency"
    $SummaryOutput = Join-Path $RepoRoot "analysis\${Campaign}_debug_results.md"
}
else {
    $TrainProject = Join-Path $RepoRoot "runs\detect\$Campaign"
    $TestProject = Join-Path $RepoRoot "runs\test\$Campaign"
    $QualityProject = Join-Path $RepoRoot "runs\quality\$Campaign"
    $CommonProject = Join-Path $RepoRoot "runs\common_eval\$Campaign"
    $LatencyProject = Join-Path $RepoRoot "runs\latency\$Campaign"
    $SummaryOutput = Join-Path $RepoRoot "analysis\${Campaign}_results.md"
}

$EffectiveEpochs = if ($Debug) { 1 } else { $Epochs }
$EffectiveBatch = if ($Debug) { 2 } else { $Batch }
$EffectiveWorkers = if ($Debug) { 0 } else { $Workers }
$EffectiveValWorkers = if ($Debug) { 0 } else { $ValWorkers }
$EffectivePatience = if ($Debug) { 1 } else { $Patience }
$Fraction = if ($Debug) { "0.02" } else { "1.0" }
$RunToken = if ($Debug) { "__debug" } else { "" }

function Assert-LeafFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing required file: $Path" }
}

function Format-CommandLine([string]$Executable, [string[]]$Arguments) {
    $quoted = @($Executable) + $Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return $quoted -join ' '
}

function Remove-VerifiedDebugCampaign([string]$Path) {
    $ExpectedParent = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "runs\debug")).TrimEnd('\', '/')
    $ResolvedTarget = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $ResolvedParent = [System.IO.Path]::GetDirectoryName($ResolvedTarget).TrimEnd('\', '/')
    $TargetLeaf = [System.IO.Path]::GetFileName($ResolvedTarget)
    if ($ResolvedParent -ne $ExpectedParent -or $TargetLeaf -ne $Campaign) {
        throw "Refusing unsafe debug cleanup target: $ResolvedTarget"
    }
    if (Test-Path -LiteralPath $ResolvedTarget -PathType Container) {
        Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force
        Write-Host "Deleted successful debug results: $ResolvedTarget" -ForegroundColor DarkCyan
    }
}

if ($Seeds.Count -eq 0) { throw "At least one seed is required." }
if (($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) { throw "Seeds must be non-negative." }
if (($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) { throw "Seeds must not contain duplicates." }
if (-not $Debug) {
    $UnexpectedSeeds = @(Compare-Object -ReferenceObject @(0, 1, 2) -DifferenceObject @($Seeds | Sort-Object))
    if ($UnexpectedSeeds.Count -ne 0) { throw "Formal confirmation is locked to seeds 0, 1, and 2." }
}
if ($Specs.Count -eq 0) { throw "At least one valid experiment is required." }
foreach ($path in @(
    $PyExe, $TrainScript, $ValScript, $QualityScript, $CommonEvalScript,
    $LatencyScript, $SummaryScript, $DatasetB, $DatasetBCommonAnnotations, $Weights
)) { Assert-LeafFile $path }
if (-not (Test-Path -LiteralPath $DatasetBTestImages -PathType Container)) {
    throw "Missing required directory: $DatasetBTestImages"
}
foreach ($spec in $Specs) { Assert-LeafFile $spec.Model }
$LatencyImage = Get-ChildItem -LiteralPath $DatasetBTestImages -File |
    Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp') } |
    Sort-Object Name | Select-Object -First 1 -ExpandProperty FullName
if (-not $LatencyImage) { throw "No latency image found in: $DatasetBTestImages" }

$TotalRuns = $Specs.Count * $Seeds.Count
$HasBothModels = @($Specs.Id | Sort-Object) -join ',' -eq 'FSDH_YOLO,YOLO11n'
Write-Host "Experiment plan" -ForegroundColor Cyan
Write-Host "Mode        : $(if ($Debug) { 'DEBUG (1 epoch, 2% fraction)' } else { 'FORMAL' })"
Write-Host "Dataset     : Dataset B v4 aug"
Write-Host "Experiments : $($Specs.Id -join ', ')"
Write-Host "Seeds       : $($Seeds -join ', ')"
Write-Host "Train runs  : $TotalRuns"
Write-Host "Pipeline    : train -> independent test -> COCO-area quality -> common evaluator$(if ($SkipLatency) { '' } else { ' -> FP32 b1 latency' }) -> Markdown comparison"
Write-Host "Epochs      : $EffectiveEpochs"
Write-Host "Campaign    : $Campaign"

if (-not $PlanOnly) {
    New-Item -ItemType Directory -Force -Path $TrainProject, $TestProject, $QualityProject, $CommonProject, $LatencyProject | Out-Null
}

foreach ($spec in $Specs) {
    foreach ($seed in $Seeds) {
        $EffectiveRunBase = "$($spec.RunBase)$RunToken"
        $RunName = "${EffectiveRunBase}__seed$seed"
        $RunDir = Join-Path $TrainProject $RunName
        $BestPt = Join-Path $RunDir "weights\best.pt"
        $FinalSummary = Join-Path $RunDir "final_summary.txt"
        $TestName = "${RunName}_test"
        $TestDir = Join-Path $TestProject $TestName
        $MetricsJson = Join-Path $TestDir "metrics_summary.json"
        $QualityJson = Join-Path $QualityProject "${RunName}_test_quality.json"
        $CommonDir = Join-Path $CommonProject $spec.CommonId
        $CommonJson = Join-Path $CommonDir "seed${seed}__test_metrics_common.json"
        $CommonPredictions = Join-Path $CommonDir "seed${seed}__test_predictions_coco.json"
        $LatencyDir = Join-Path $LatencyProject $RunName
        $LatencyJson = Join-Path $LatencyDir "uniform_latency_results.json"
        $LatencyManifest = Join-Path $LatencyDir "manifest.csv"

        $TrainArgs = @(
            "-u", $TrainScript, "--model", $spec.Model, "--data", $DatasetB,
            "--project", $TrainProject, "--name", $RunName, "--epochs", [string]$EffectiveEpochs,
            "--batch", [string]$EffectiveBatch, "--workers", [string]$EffectiveWorkers,
            "--val-workers", [string]$EffectiveValWorkers, "--patience", [string]$EffectivePatience,
            "--seed", [string]$seed, "--fraction", $Fraction, "--iou-type", "ciou"
        )
        $TrainArgs += @($spec.TrainInitArgs)
        $TestArgs = @(
            "-u", $ValScript, "--model", $BestPt, "--data", $DatasetB,
            "--batch", [string]$ValBatch, "--project", $TestProject, "--name", $TestName
        )
        $QualityArgs = @(
            "-u", $QualityScript, "--model", $BestPt, "--data", $DatasetB,
            "--split", "test", "--imgsz", "640", "--batch", "16", "--device", "0",
            "--conf", "0.001", "--iou", "0.7", "--max-det", "300", "--bins", "15",
            "--area-scale", "1", "--small-metric-name", "AP_small",
            "--class1-name", "half", "--out", $QualityJson
        )
        $CommonArgs = @(
            "-u", $CommonEvalScript, "--model", $BestPt, "--model-name", $spec.Label,
            "--seed", [string]$seed, "--images", $DatasetBTestImages,
            "--annotations", $DatasetBCommonAnnotations, "--out", $CommonJson,
            "--predictions-out", $CommonPredictions, "--imgsz", "640", "--device", "0",
            "--conf", "0.001", "--iou", "0.7", "--max-det", "300", "--bins", "15",
            "--area-scale", "1"
        )
        $LatencyArgs = @(
            "-u", $LatencyScript, "--manifest", $LatencyManifest, "--image", $LatencyImage,
            "--out", $LatencyDir, "--imgsz", "640", "--warmup", "100", "--iterations", "500"
        )

        Write-Host ""
        Write-Host "[$($spec.Label)][seed=$seed]" -ForegroundColor Green
        Write-Host "TRAIN   $(Format-CommandLine $PyExe $TrainArgs)" -ForegroundColor DarkGray
        Write-Host "TEST    $(Format-CommandLine $PyExe $TestArgs)" -ForegroundColor DarkGray
        Write-Host "QUALITY $(Format-CommandLine $PyExe $QualityArgs)" -ForegroundColor DarkGray
        Write-Host "COMMON  $(Format-CommandLine $PyExe $CommonArgs)" -ForegroundColor DarkGray
        if (-not $SkipLatency) { Write-Host "LATENCY $(Format-CommandLine $PyExe $LatencyArgs)" -ForegroundColor DarkGray }
        if ($PlanOnly) { continue }

        if ((Test-Path $BestPt -PathType Leaf) -and (Test-Path $FinalSummary -PathType Leaf)) {
            Write-Host "[SKIP TRAIN] complete: $RunName" -ForegroundColor DarkCyan
        }
        else {
            if (Test-Path $RunDir) { throw "Partial training directory exists; inspect it: $RunDir" }
            & $PyExe @TrainArgs
            $exit = $LASTEXITCODE
            if (-not (Test-Path $BestPt -PathType Leaf) -or -not (Test-Path $FinalSummary -PathType Leaf)) {
                throw "Training failed or incomplete (exit=$exit): $RunName"
            }
        }

        if (-not (Test-Path $MetricsJson -PathType Leaf)) {
            if (Test-Path $TestDir) { throw "Partial test directory exists; inspect it: $TestDir" }
            & $PyExe @TestArgs
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $MetricsJson -PathType Leaf)) { throw "Test failed: $RunName" }
        }

        $QualityCurrent = $false
        if (Test-Path $QualityJson -PathType Leaf) {
            $QualityPayload = Get-Content -LiteralPath $QualityJson -Raw | ConvertFrom-Json
            $QualityCurrent = $QualityPayload.size_ap_protocol.name -eq "COCOeval area-range ignore semantics"
        }
        if (-not $QualityCurrent) {
            & $PyExe @QualityArgs
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $QualityJson -PathType Leaf)) { throw "Quality failed: $RunName" }
        }

        if (-not (Test-Path $CommonJson -PathType Leaf)) {
            New-Item -ItemType Directory -Force -Path $CommonDir | Out-Null
            & $PyExe @CommonArgs
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $CommonJson -PathType Leaf)) { throw "Common evaluation failed: $RunName" }
        }

        if (-not $SkipLatency -and -not (Test-Path $LatencyJson -PathType Leaf)) {
            New-Item -ItemType Directory -Force -Path $LatencyDir | Out-Null
            @([pscustomobject]@{
                section = $spec.Id; model = $spec.Label; seed = $seed; weights = $BestPt
            }) | Export-Csv -LiteralPath $LatencyManifest -NoTypeInformation -Encoding utf8
            & $PyExe @LatencyArgs
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $LatencyJson -PathType Leaf)) { throw "Latency failed: $RunName" }
        }
    }
}

if ($HasBothModels) {
    $BaselineSpec = $AllSpecs | Where-Object Id -eq "YOLO11n"
    $FsdhSpec = $AllSpecs | Where-Object Id -eq "FSDH_YOLO"
    $SummaryArgs = @(
        "-u", $SummaryScript, "--common-root", $CommonProject,
        "--train-root", $TrainProject, "--test-root", $TestProject,
        "--output", $SummaryOutput,
        "--baseline-run-base", "$($BaselineSpec.RunBase)$RunToken",
        "--fsdh-run-base", "$($FsdhSpec.RunBase)$RunToken",
        "--report-title", "Dataset B v4 aug：YOLO11n 与 FSDH-YOLO 对比结果",
        "--dataset-version", "v4_aug", "--seeds"
    )
    $SummaryArgs += @($Seeds | ForEach-Object { [string]$_ })
    if (-not $SkipLatency) { $SummaryArgs += @("--latency-root", $LatencyProject) }
    Write-Host ""
    Write-Host "SUMMARY $(Format-CommandLine $PyExe $SummaryArgs)" -ForegroundColor DarkGray
    if (-not $PlanOnly) {
        & $PyExe @SummaryArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $SummaryOutput -PathType Leaf)) {
            throw "Comparison report generation failed: $SummaryOutput"
        }
        Write-Host "Comparison report: $SummaryOutput" -ForegroundColor Green
    }
}
else {
    Write-Host "Comparison report skipped because both YOLO11n and FSDH_YOLO were not selected." -ForegroundColor Yellow
}

Write-Host ""
if ($PlanOnly) {
    Write-Host "Plan only: $TotalRuns training runs listed; nothing was started or deleted." -ForegroundColor Cyan
}
elseif ($Debug) {
    Remove-VerifiedDebugCampaign $DebugCampaignRoot
    Write-Host "Debug pipelines completed; the Markdown report was retained under analysis." -ForegroundColor Cyan
}
else {
    Write-Host "Selected formal pipelines and comparison report completed." -ForegroundColor Cyan
}
