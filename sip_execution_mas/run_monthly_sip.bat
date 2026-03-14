@echo off
REM Monthly SIP Execution — triggered by Windows Task Scheduler on the 1st of each month
REM Logs output to sip_execution_mas\outputs\scheduler.log

set PYTHONPATH=C:\Users\Admin\fin-agents;C:\Users\Admin\fin-agents\sip_execution_mas
set PYTHONUTF8=1

REM Load GEMINI_API_KEY from .env (simple parse)
for /f "tokens=1,2 delims==" %%A in ('findstr /i "GEMINI_API_KEY" "C:\Users\Admin\fin-agents\.env"') do (
    set %%A=%%B
)

set LOG_FILE=C:\Users\Admin\fin-agents\sip_execution_mas\outputs\scheduler.log

echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo [%DATE% %TIME%] Monthly SIP triggered >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

C:\Python314\python.exe -m simulator.scheduler --now --sip 500 >> "%LOG_FILE%" 2>&1

echo [%DATE% %TIME%] Done >> "%LOG_FILE%"
