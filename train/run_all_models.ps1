<#
.SYNOPSIS
Runs the frozen Dataset B comparison: YOLO11n versus FSDH-YOLO-n.

.DESCRIPTION
The formal protocol is locked to Dataset B groupaware_clean_v4_aug and seeds
0/1/2. Each best.pt is evaluated on the frozen test split twice: Ultralytics
test for checkpoint/complexity traceability and the framework-neutral Common
Evaluator for every reported accuracy metric.

The Common Evaluator is the only source for P/R/mAP, AP75, COCO AP-small, and
per-maturity AP50/AP50-95. Dataset B uses standard COCO AP-small at the original
640x640 image scale (area < 32^2 pixels), not Dataset A's area-times-four metric.

Examples:
  .\run_all_models.ps1 -PlanOnly
  .\run_all_models.ps1
  .\run_all_models.ps1 -Debug
#>

param(
    [ValidateRange(1, 10000)][int]$Epochs = 300,
    [ValidateRange(1, 1024)][int]$Batch = 12,
    [ValidateRange(0, 8)][int]$Workers = 6,
    [ValidateRange(0, 4)][int]$ValWorkers = 2,
    [ValidateRange(1, 1024)][int]$TestBatch = 16,
    [ValidateRange(1, 10000)][int]$Patience = 100,
    [int[]]$Seeds = @(0, 1, 2),
    [ValidateSet("YOLO11n", "FSDH_YOLO")]
    [string[]]$Experiments = @("YOLO11n", "FSDH_YOLO"),
    [ValidatePattern("^[A-Za-z0-9_.-]+$")][string]$RunTag = "main",
    [string]$Device = "0",
    [switch]$QuietProgress,
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
$PreferredPython = "<python>"
$PyExe = if (Test-Path -LiteralPath $PreferredPython -PathType Leaf) {
    $PreferredPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

$DatasetRoot = "<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_aug"
$DatasetRawRoot = "<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_raw"
$DataYaml = Join-Path $DatasetRoot "lichi_maturity_groupaware_clean_v4_aug.yaml"
$TestImages = Join-Path $DatasetRoot "test\images"
$CocoAnnotations = Join-Path $RepoRoot "external_configs\dataset_b_native4_v4\instances_test.json"
$ExportCocoScript = Join-Path $RepoRoot "scripts\export_yolo_split_to_coco.py"
$TrainScript = Join-Path $RepoRoot "origin_train.py"
$TestScript = Join-Path $RepoRoot "val.py"
$CommonEvalScript = Join-Path $RepoRoot "scripts\eval_ultralytics_common_evaluator.py"
$SummaryScript = Join-Path $RepoRoot "scripts\summarize_dataset_b_common_evaluator.py"
$Weights = Join-Path $RepoRoot "yolo11n.pt"

# Formal execution refuses to start if any cleaning, split, augmentation, or
# independent-verification record is missing.
$ProvenanceFiles = @(
    (Join-Path $DatasetRawRoot "audit\protocol_lock.json"),
    (Join-Path $DatasetRawRoot "audit\review_decisions.json"),
    (Join-Path $DatasetRawRoot "audit\box_review_manifest.json"),
    (Join-Path $DatasetRawRoot "build_summary.json"),
    (Join-Path $DatasetRawRoot "group_manifest.csv"),
    (Join-Path $DatasetRawRoot "image_constraint_edges.csv"),
    (Join-Path $DatasetRawRoot "label_cleaning_manifest.csv"),
    (Join-Path $DatasetRawRoot "split_manifest.csv"),
    (Join-Path $DatasetRoot "raw_build_summary.json"),
    (Join-Path $DatasetRoot "augmentation_summary.json"),
    (Join-Path $DatasetRoot "augmentation_log.csv"),
    (Join-Path $RepoRoot "analysis\dataset_b_v4_verification.json")
)

$AllSpecs = @(
    [pscustomobject]@{
        Id = "YOLO11n"
        Label = "YOLO11n"
        CommonId = "DatasetB4_YOLO11n"
        Model = $Weights
        RunBase = "dsB_groupaware_clean_v4_aug__comparison__yolo11n"
        TrainInitArgs = @()
    },
    [pscustomobject]@{
        Id = "FSDH_YOLO"
        Label = "FSDH-YOLO-n"
        CommonId = "DatasetB4_FSDH_YOLO"
        Model = Join-Path $RepoRoot "ultralytics\cfg\models\11\yolo11n_FSDConv-SHAB.yaml"
        RunBase = "dsB_groupaware_clean_v4_aug__comparison__fsdh_yolo_n"
        TrainInitArgs = @("--weights", $Weights, "--weights-map", "none")
    }
)
$Specs = @($AllSpecs | Where-Object { $_.Id -in $Experiments })

$Campaign = "paper_dataset_b_groupaware_clean_v4_aug_common_v2"
$RunSuffix = if ($Debug) { "__debug__$RunTag" } elseif ($RunTag -eq "main") { "" } else { "__$RunTag" }
if ($Debug) {
    $CampaignRoot = Join-Path $RepoRoot "runs\debug\$Campaign"
    $TrainProject = Join-Path $CampaignRoot "train"
    $TestProject = Join-Path $CampaignRoot "test"
    $CommonProject = Join-Path $CampaignRoot "common_eval"
    $SummaryOutput = Join-Path $RepoRoot "analysis\${Campaign}__debug__${RunTag}_results.md"
} else {
    $TrainProject = Join-Path $RepoRoot "runs\detect\$Campaign"
    $TestProject = Join-Path $RepoRoot "runs\test\$Campaign"
    $CommonProject = Join-Path $RepoRoot "runs\common_eval\$Campaign"
    $SummaryOutput = Join-Path $RepoRoot "analysis\${Campaign}__${RunTag}_results.md"
}

$EffectiveEpochs = if ($Debug) { 1 } else { $Epochs }
$EffectiveBatch = if ($Debug) { 2 } else { $Batch }
$EffectiveWorkers = if ($Debug) { 0 } else { $Workers }
$EffectiveValWorkers = if ($Debug) { 0 } else { $ValWorkers }
$EffectivePatience = if ($Debug) { 1 } else { $Patience }
$EffectiveSeeds = if ($Debug) { @($Seeds[0]) } else { @($Seeds) }
$Fraction = if ($Debug) { "0.02" } else { "1.0" }

function Assert-LeafFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing $Label`: $Path" }
}

function Format-CommandLine([string]$Executable, [string[]]$Arguments) {
    $quoted = @($Executable) + $Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return $quoted -join ' '
}

function Invoke-Python([string[]]$Arguments) {
    Write-Host "> $(Format-CommandLine $PyExe $Arguments)" -ForegroundColor DarkGray
    & $PyExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

function Assert-FrozenTestAnnotations {
    Assert-LeafFile $CocoAnnotations "current test COCO annotations"
    $payload = Get-Content -LiteralPath $CocoAnnotations -Raw | ConvertFrom-Json
    if ($payload.info.split -ne "test") { throw "COCO annotation file is not marked as test: $CocoAnnotations" }
    if ($payload.images.Count -ne 212 -or $payload.annotations.Count -ne 3263) {
        throw "Frozen test mismatch: expected 212 images/3263 annotations, got $($payload.images.Count)/$($payload.annotations.Count)"
    }
    $categoryNames = @($payload.categories | Sort-Object id | ForEach-Object { $_.name })
    if (($categoryNames -join ',') -ne "green,half,red,young") {
        throw "Unexpected test categories: $($categoryNames -join ',')"
    }
    return (Get-FileHash -LiteralPath $CocoAnnotations -Algorithm SHA256).Hash.ToLowerInvariant()
}

if ($Seeds.Count -eq 0) { throw "At least one seed is required." }
if (($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) { throw "Seeds must be non-negative." }
if (($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) { throw "Seeds must not contain duplicates." }
if (-not $Debug) {
    $UnexpectedSeeds = @(Compare-Object -ReferenceObject @(0, 1, 2) -DifferenceObject @($Seeds | Sort-Object))
    if ($UnexpectedSeeds.Count -ne 0) { throw "Formal comparison is locked to seeds 0, 1, and 2." }
}
if ($Specs.Count -eq 0) { throw "At least one valid experiment is required." }

foreach ($item in @(
    @($PyExe, "Python interpreter"),
    @($DataYaml, "Dataset B v4 augmented YAML"),
    @($ExportCocoScript, "YOLO-to-COCO exporter"),
    @($TrainScript, "training script"),
    @($TestScript, "independent test script"),
    @($CommonEvalScript, "Common Evaluator adapter"),
    @($SummaryScript, "three-seed comparison summarizer"),
    @($Weights, "YOLO11n pretrained checkpoint")
)) { Assert-LeafFile $item[0] $item[1] }
foreach ($path in $ProvenanceFiles) { Assert-LeafFile $path "retained data-provenance artifact" }
foreach ($spec in $Specs) { Assert-LeafFile $spec.Model "$($spec.Label) model" }
if (-not (Test-Path -LiteralPath $TestImages -PathType Container)) { throw "Missing frozen test images: $TestImages" }

$VerificationPath = Join-Path $RepoRoot "analysis\dataset_b_v4_verification.json"
$Verification = Get-Content -LiteralPath $VerificationPath -Raw | ConvertFrom-Json
if ($Verification.status -ne "PASS") { throw "Dataset B v4 independent verification is not PASS." }
if ($Verification.raw_statistics.test.images -ne 212 -or $Verification.raw_statistics.test.objects -ne 3263) {
    throw "Dataset verification does not match the frozen test population."
}

$ExportArgs = @("-u", $ExportCocoScript, "--data", $DataYaml, "--split", "test", "--out", $CocoAnnotations)
if ($PlanOnly) {
    $AnnotationSha256 = Assert-FrozenTestAnnotations
} else {
    # Re-export from current labels before every execution, preventing a stale
    # GT snapshot from being used silently.
    Invoke-Python $ExportArgs
    $AnnotationSha256 = Assert-FrozenTestAnnotations
}

$TotalRuns = $Specs.Count * $EffectiveSeeds.Count
Write-Host "=== Dataset B: YOLO11n vs FSDH-YOLO-n ===" -ForegroundColor Cyan
Write-Host "Mode          : $(if ($Debug) { 'DEBUG (1 epoch, 2% train fraction)' } else { 'FORMAL' })"
Write-Host "Dataset       : lichi_701515_groupaware_clean_v4_aug"
Write-Host "Test GT       : 212 images / 3263 instances"
Write-Host "Test GT SHA256: $AnnotationSha256"
Write-Host "Models        : $($Specs.Label -join ', ')"
Write-Host "Seeds         : $($EffectiveSeeds -join ', ')"
Write-Host "Train runs    : $TotalRuns"
Write-Host "Evaluator     : Common Evaluator; split=test; imgsz=640; conf=0.001; NMS IoU=0.7; max_det=300"
Write-Host "Special tasks : AP75 | AP_small_COCO (<1024 px^2 at native 640) | per-maturity AP50/AP50-95"
Write-Host "Outputs       : $CommonProject"
Write-Host "COCO EXPORT   : $(Format-CommandLine $PyExe $ExportArgs)" -ForegroundColor DarkGray

if (-not $PlanOnly) {
    & $PyExe -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) { throw "CUDA is not visible to $PyExe. Use -PlanOnly to inspect the campaign." }
    New-Item -ItemType Directory -Force -Path $TrainProject, $TestProject, $CommonProject | Out-Null
}

foreach ($spec in $Specs) {
    foreach ($seed in $EffectiveSeeds) {
        $RunBase = "$($spec.RunBase)$RunSuffix"
        $RunName = "${RunBase}__seed$seed"
        $RunDir = Join-Path $TrainProject $RunName
        $BestPt = Join-Path $RunDir "weights\best.pt"
        $FinalSummary = Join-Path $RunDir "final_summary.txt"
        $InitializationSummary = Join-Path $RunDir "initialization_summary.json"
        $TestName = "${RunName}_test"
        $TestDir = Join-Path $TestProject $TestName
        $TestMetrics = Join-Path $TestDir "metrics_summary.json"
        $CommonDir = Join-Path $CommonProject $spec.CommonId
        $CommonJson = Join-Path $CommonDir "seed${seed}__test_metrics_common.json"
        $CommonPredictions = Join-Path $CommonDir "seed${seed}__test_predictions_coco.json"

        $TrainArgs = @(
            "-u", $TrainScript, "--model", $spec.Model, "--data", $DataYaml,
            "--project", $TrainProject, "--name", $RunName, "--epochs", [string]$EffectiveEpochs,
            "--batch", [string]$EffectiveBatch, "--imgsz", "640", "--workers", [string]$EffectiveWorkers,
            "--val-workers", [string]$EffectiveValWorkers, "--patience", [string]$EffectivePatience,
            "--seed", [string]$seed, "--fraction", $Fraction, "--iou-type", "ciou"
        )
        $TrainArgs += @($spec.TrainInitArgs)
        $TestArgs = @(
            "-u", $TestScript, "--model", $BestPt, "--data", $DataYaml,
            "--project", $TestProject, "--name", $TestName,
            "--batch", [string]$TestBatch, "--imgsz", "640"
        )
        $CommonArgs = @(
            "-u", $CommonEvalScript, "--model", $BestPt, "--model-name", $spec.Label,
            "--seed", [string]$seed, "--images", $TestImages, "--annotations", $CocoAnnotations,
            "--out", $CommonJson, "--predictions-out", $CommonPredictions,
            "--imgsz", "640", "--device", $Device, "--conf", "0.001", "--iou", "0.7",
            "--max-det", "300", "--bins", "15", "--area-scale", "1"
        )

        Write-Host ""
        Write-Host "[$($spec.Label)][seed=$seed]" -ForegroundColor Green
        Write-Host "TRAIN  $(Format-CommandLine $PyExe $TrainArgs)" -ForegroundColor DarkGray
        Write-Host "TEST   $(Format-CommandLine $PyExe $TestArgs)" -ForegroundColor DarkGray
        Write-Host "COMMON $(Format-CommandLine $PyExe $CommonArgs)" -ForegroundColor DarkGray
        if ($PlanOnly) { continue }

        if ((Test-Path -LiteralPath $BestPt -PathType Leaf) -and
            (Test-Path -LiteralPath $FinalSummary -PathType Leaf) -and
            (Test-Path -LiteralPath $InitializationSummary -PathType Leaf)) {
            Write-Host "[REUSE TRAIN] $RunName" -ForegroundColor DarkCyan
        } else {
            if (Test-Path -LiteralPath $RunDir) {
                throw "Partial training directory exists; inspect it or choose another -RunTag: $RunDir"
            }
            Invoke-Python $TrainArgs
            Assert-LeafFile $BestPt "best checkpoint"
            Assert-LeafFile $FinalSummary "training final summary"
            Assert-LeafFile $InitializationSummary "initialization audit"
        }

        if (Test-Path -LiteralPath $TestMetrics -PathType Leaf) {
            $TestPayload = Get-Content -LiteralPath $TestMetrics -Raw | ConvertFrom-Json
            if ($TestPayload.split -ne "test" -or
                [System.IO.Path]::GetFullPath($TestPayload.model) -ne [System.IO.Path]::GetFullPath($BestPt)) {
                throw "Existing official test result is not for this best.pt/test split: $TestMetrics"
            }
            Write-Host "[REUSE TEST] $TestName" -ForegroundColor DarkCyan
        } else {
            if (Test-Path -LiteralPath $TestDir) {
                throw "Partial test directory exists; inspect it or choose another -RunTag: $TestDir"
            }
            Invoke-Python $TestArgs
            Assert-LeafFile $TestMetrics "official independent-test metrics"
        }

        if (Test-Path -LiteralPath $CommonJson -PathType Leaf) {
            $CommonPayload = Get-Content -LiteralPath $CommonJson -Raw | ConvertFrom-Json
            $CommonCurrent = (
                $CommonPayload.split -eq "test" -and
                $CommonPayload.seed -eq $seed -and
                $CommonPayload.ground_truth.sha256 -eq $AnnotationSha256 -and
                $CommonPayload.ground_truth.images -eq 212 -and
                $CommonPayload.ground_truth.annotations -eq 3263 -and
                $CommonPayload.protocol.small_metric_name -eq "AP_small_COCO" -and
                $CommonPayload.protocol.small_area_scale -eq 1.0
            )
            if (-not $CommonCurrent) {
                throw "Existing Common Evaluator result uses a stale/different protocol; choose another -RunTag: $CommonJson"
            }
            Write-Host "[REUSE COMMON] $CommonJson" -ForegroundColor DarkCyan
        } else {
            New-Item -ItemType Directory -Force -Path $CommonDir | Out-Null
            Invoke-Python $CommonArgs
            Assert-LeafFile $CommonJson "Common Evaluator test metrics"
            Assert-LeafFile $CommonPredictions "Common Evaluator COCO predictions"
        }
    }
}

$HasBothModels = @($Specs.Id | Sort-Object) -join ',' -eq 'FSDH_YOLO,YOLO11n'
if ($HasBothModels) {
    $BaselineSpec = $AllSpecs | Where-Object Id -eq "YOLO11n"
    $FsdhSpec = $AllSpecs | Where-Object Id -eq "FSDH_YOLO"
    $SummaryArgs = @(
        "-u", $SummaryScript, "--common-root", $CommonProject,
        "--train-root", $TrainProject, "--test-root", $TestProject,
        "--output", $SummaryOutput,
        "--baseline-run-base", "$($BaselineSpec.RunBase)$RunSuffix",
        "--fsdh-run-base", "$($FsdhSpec.RunBase)$RunSuffix",
        "--report-title", "Dataset B v4 aug：YOLO11n 与 FSDH-YOLO-n Common Evaluator 对比",
        "--dataset-version", "v4_aug_common_v2", "--seeds"
    )
    $SummaryArgs += @($EffectiveSeeds | ForEach-Object { [string]$_ })
    Write-Host ""
    Write-Host "SUMMARY $(Format-CommandLine $PyExe $SummaryArgs)" -ForegroundColor DarkGray
    if (-not $PlanOnly) {
        Invoke-Python $SummaryArgs
        Assert-LeafFile $SummaryOutput "comparison report"
        Assert-LeafFile ($SummaryOutput -replace '\.md$', '.json') "comparison JSON sidecar"
        Assert-LeafFile ($SummaryOutput -replace '\.md$', '_task_ap75.csv') "AP75 task result"
        Assert-LeafFile ($SummaryOutput -replace '\.md$', '_task_apsmall_coco.csv') "COCO AP-small task result"
        Assert-LeafFile ($SummaryOutput -replace '\.md$', '_task_maturity_ap.csv') "maturity AP task result"
    }
} else {
    Write-Host "Comparison summary skipped because both models were not selected." -ForegroundColor Yellow
}

Write-Host ""
if ($PlanOnly) {
    Write-Host "Plan only: $TotalRuns runs listed; nothing was trained, evaluated, overwritten, or deleted." -ForegroundColor Cyan
} else {
    Write-Host "Completed Dataset B test comparison with Common Evaluator." -ForegroundColor Green
    Write-Host "Train results : $TrainProject"
    Write-Host "Test results  : $TestProject"
    Write-Host "Common results: $CommonProject"
    if ($HasBothModels) { Write-Host "Summary       : $SummaryOutput" }
}
