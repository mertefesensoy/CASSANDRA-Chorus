<#
.SYNOPSIS
  Start one CASSANDRA Chorus command in a visible PowerShell window.

.DESCRIPTION
  Integrates the parent CASSANDRA project's visible-terminal rule (ADR 0012):
  evidence-producing runs are started where they can be watched. The new window
  opens at the repository root, optionally sets CHORUS_RUNS_DIR, runs
  "python -m <Module> <Arguments>", and stays open with the exit code shown.
  Everything the window shows is also saved as a transcript (default:
  <RunsDir or repository>\launcher_logs\<time>_<module>.log), the equivalent of
  the legacy launcher log, so the run can be reviewed afterwards.
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
    [string]$RunsDir = "",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($Module -notmatch '^[A-Za-z0-9_.]+$') { throw "Module must be a dotted Python module name, got '$Module'" }

function Quote([string]$Text) { "'" + $Text.Replace("'", "''") + "'" }

if ($LogPath -eq "") {
    $LogRoot = if ($RunsDir -ne "") { $RunsDir } else { $RepoRoot }
    $LogPath = Join-Path (Join-Path $LogRoot "launcher_logs") ("{0}_{1}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"), $Module)
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

# The window runs a generated script file rather than a -Command string:
# Start-Process does not quote argument-list items containing spaces, which
# mangled paths such as the repository's (found 2026-10-08). The script is kept
# beside the transcript as a record of exactly what was launched.
$ScriptPath = [IO.Path]::ChangeExtension($LogPath, ".launch.ps1")
# One statement per line: in PowerShell the comma binds tighter than "+", so
# building this list with @("a" + $x, "b") silently nests arrays (found 2026-10-08).
$Lines = New-Object System.Collections.Generic.List[string]
$Lines.Add("Start-Transcript -LiteralPath " + (Quote $LogPath) + " | Out-Null")
$Lines.Add("`$Host.UI.RawUI.WindowTitle = " + (Quote "Chorus: $Module"))
$Lines.Add("Set-Location -LiteralPath " + (Quote $RepoRoot))
if ($RunsDir -ne "") { $Lines.Add("`$env:CHORUS_RUNS_DIR = " + (Quote $RunsDir)) }
$Lines.Add("Write-Host ('[chorus] ' + (Get-Date -Format o) + ' start: python -m $Module ' + " + (Quote $Arguments) + ")")
$Lines.Add("python -m $Module $Arguments")
$Lines.Add("`$code = `$LASTEXITCODE")
$Lines.Add("Write-Host ('[chorus] ' + (Get-Date -Format o) + ' exit code ' + `$code)")
$Lines.Add("Stop-Transcript | Out-Null")
# UTF-8 with byte-order mark, so Windows PowerShell 5.1 reads non-ASCII paths correctly.
[IO.File]::WriteAllText($ScriptPath, ($Lines -join "`r`n") + "`r`n", (New-Object System.Text.UTF8Encoding $true))
$Process = Start-Process -FilePath "powershell.exe" -WorkingDirectory $RepoRoot -PassThru `
    -ArgumentList ('-NoExit -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $ScriptPath)
Write-Host "[chorus] started visible window, process id $($Process.Id): python -m $Module $Arguments"
Write-Host "[chorus] transcript: $LogPath"
Write-Host "[chorus] launch script: $ScriptPath"
