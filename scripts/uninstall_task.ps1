# Stops and removes the "Sentinel" scheduled task.
Stop-ScheduledTask -TaskName "Sentinel" -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "Sentinel" -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Task 'Sentinel' removed."
