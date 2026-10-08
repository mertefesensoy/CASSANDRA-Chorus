<#
.SYNOPSIS
  Start one CASSANDRA Chorus command in a visible PowerShell window.

.DESCRIPTION
  Integrates the parent CASSANDRA project's visible-terminal rule (ADR 0012):
  evidence-producing runs are started where they can be watched. The new window
  opens at the repository root, optionally sets CHORUS_RUNS_DIR, runs
  "python -m <Module> <Arguments>", and stays open with the exit code shown.
  Keep-awake, power logging and pre-start checks happen inside the Python run
  itself (cassandra_chorus.ops), so they apply however a run is started.
  See docs/implementations/2026-10-08-run-operations.md.

.EXAMPLE
  powershell -File scripts\ops\launch_visible.ps1 -Module scripts.train_task_a_centralized `
    -Arguments "--config configs/task_a/centralized.toml --set run.seed=11" -RunsDir C:\Users\senso\chorus-runs
#>
param(
    [Parameter(Mandatory = $true)][string]$Module,
    [string]$Arguments = "",
    [string]$RunsDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($Module -notmatch '^[A-Za-z0-9_.]+$') { throw "Module must be a dotted Python module name, got '$Module'" }

function Quote([string]$Text) { "'" + $Text.Replace("'", "''") + "'" }

$Lines = @(
    "`$Host.UI.RawUI.WindowTitle = " + (Quote "Chorus: $Module"),
    "Set-Location -LiteralPath " + (Quote $RepoRoot)
)
if ($RunsDir -ne "") { $Lines += "`$env:CHORUS_RUNS_DIR = " + (Quote $RunsDir) }
$Lines += @(
    "Write-Host ('[chorus] ' + (Get-Date -Format o) + ' start: python -m $Module ' + " + (Quote $Arguments) + ")",
    "python -m $Module $Arguments",
    "`$code = `$LASTEXITCODE",
    "Write-Host ('[chorus] ' + (Get-Date -Format o) + ' exit code ' + `$code)"
)
$Command = $Lines -join "; "
$Process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-NoProfile", "-Command", $Command) -WorkingDirectory $RepoRoot -PassThru
Write-Host "[chorus] started visible window, process id $($Process.Id): python -m $Module $Arguments"
