# Registers a Windows Scheduled Task "Sentinel" that starts the engine at logon and restarts it if it exits.
# Run from an elevated PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
$root = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $root "scripts\run.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`" run" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Sentinel" -Action $action -Trigger $trigger -Settings $settings -Description "Sentinel v2 trading lab" -Force | Out-Null
Start-ScheduledTask -TaskName "Sentinel"
Write-Host "Task 'Sentinel' registered and started. Dashboard: http://127.0.0.1:8787  Logs: $root\logs"
