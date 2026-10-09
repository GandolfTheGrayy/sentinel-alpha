# Runs Sentinel and keeps it running (used by the scheduled task; also fine to run by hand).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run.ps1 [run|doctor|...]
# Logs go to logs\sentinel-YYYY-MM-DD.log (UTF-8). A crash restarts the engine after 30 s; a clean exit stops the loop.
param([string[]]$Cmd = @("run"))
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

while ($true) {
    $log = Join-Path $root ("logs\sentinel-" + (Get-Date -Format "yyyy-MM-dd") + ".log")
    "[" + (Get-Date -Format "u") + "] starting: sentinel " + ($Cmd -join " ") | Out-File -FilePath $log -Append -Encoding utf8
    & $py -m sentinel @Cmd 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; $line } | Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "[" + (Get-Date -Format "u") + "] sentinel exited with code $code" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -eq 0 -or $Cmd[0] -ne "run") { break }
    Start-Sleep -Seconds 30
}
