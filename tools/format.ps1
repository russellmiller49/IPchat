<#!
.SYNOPSIS
  Format Python files using ruff and black.

.DESCRIPTION
  Formats only changed Python files relative to HEAD. If none are detected,
  falls back to formatting key directories (tools, tests, utils, ipchat).

.PARAMETER All
  When provided, formats key directories regardless of git status.

.EXAMPLE
  ./tools/format.ps1

.EXAMPLE
  ./tools/format.ps1 -All

#>
param(
  [switch]$All
)

function Invoke-OrFail($cmd, $args) {
  Write-Host "=> $cmd $args"
  & $cmd $args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $cmd $args"
  }
}

try {
  $ErrorActionPreference = 'Stop'

  if (-not $All) {
    $changed = git diff --name-only --diff-filter=ACMRTUXB HEAD 2>$null |
      Where-Object { $_ -match '\.py$' }
  } else {
    $changed = @()
  }

  if ($changed -and $changed.Count -gt 0) {
    Write-Host "Formatting changed files:`n$($changed -join "`n")" -ForegroundColor Cyan
    Invoke-OrFail ruff "check --fix --no-cache $($changed -join ' ')"
    Invoke-OrFail python "-m black $($changed -join ' ')"
  } else {
    Write-Host "No changed Python files detected (or -All set). Formatting key dirs..." -ForegroundColor Yellow
    $targets = @('tools','tests','utils','ipchat') | Where-Object { Test-Path $_ }
    if ($targets.Count -eq 0) {
      Write-Host "No target directories found; nothing to format." -ForegroundColor DarkYellow
      exit 0
    }
    Invoke-OrFail ruff "check --fix --no-cache $($targets -join ' ')"
    Invoke-OrFail python "-m black $($targets -join ' ')"
  }

  Write-Host "Formatting complete." -ForegroundColor Green
  exit 0
}
catch {
  Write-Error $_
  Write-Host "Tip: Ensure ruff and black are installed in your environment (pip install ruff black)." -ForegroundColor DarkYellow
  exit 1
}

