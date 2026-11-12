# Registers the watchdog in Windows Task Scheduler (every 30 minutes).
# Run in a normal PowerShell window as your own user.

$TaskName     = "AGGR Collector Watchdog"
$BatPath      = "C:\aggr-subminute-collector\watchdog\run_watchdog.bat"
$EveryMinutes = 30

if (-not (Test-Path $BatPath)) {
    throw "Watchdog BAT not found: $BatPath"
}

# Run the BAT through cmd.exe.
$action = New-ScheduledTaskAction `
    -Execute "$env:ComSpec" `
    -Argument "/c `"$BatPath`"" `
    -WorkingDirectory (Split-Path $BatPath)

# Repeat indefinitely every 30 minutes.
# An omitted RepetitionDuration means no expiration.
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# S4U = run whether logged in or not, without storing your password.
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel Limited

# Stop on registration failure; do not print a false success message.
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force `
    -Description "Dispatches the AGGR collector GitHub workflow only when no successful run finished recently." `
    -ErrorAction Stop

# Verify that Windows actually registered the task.
$registeredTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop

Write-Host ""
Write-Host "SUCCESS: Task registered."
Write-Host "Task name: $($registeredTask.TaskName)"
Write-Host "Task state: $($registeredTask.State)"
Write-Host ""
Write-Host "Test:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Check: Get-ScheduledTaskInfo -TaskName '$TaskName'"