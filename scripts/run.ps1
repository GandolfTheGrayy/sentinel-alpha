# Starts Sentinel in the foreground (used by the scheduled task). Logs to logs\sentinel-YYYY-MM-DD.log
param([string[]]$Args = @("run"))
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$log = Join-Path $root ("logs\sentinel-" + (Get-Date -Format "yyyy-MM-dd") + ".log")
& $py -m sentinel @Args 2>&1 | Tee-Object -FilePath $log -Append
