@echo off
setlocal
rem Trigger only. No collector code, no local data processing.
py -3.14 "%~dp0dispatch_watchdog.py" %*
exit /b %ERRORLEVEL%