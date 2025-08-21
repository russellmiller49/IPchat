@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Timestamped log setup ---
for /f %%I in ('powershell -NoProfile -Command "(Get-Date -Format ''yyyy-MM-dd_HHmmss'')"') do set TS=%%I
set LOGDIR=logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\run_%TS%.txt

echo ============================================== > "%LOGFILE%"
echo  Textbook Batch Run - %DATE% %TIME%             >> "%LOGFILE%"
echo  Project: %CD%                                  >> "%LOGFILE%"
echo ============================================== >> "%LOGFILE%"

REM --- Activate conda env (IPchat) ---
REM If conda is already initialized for cmd.exe, the next line is enough:
call conda activate IPchat >nul 2>&1
if errorlevel 1 (
  REM Fallback: call the activate script directly (adjust path if needed)
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
  call conda activate IPchat
)

where conda >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Conda not found on PATH. >> "%LOGFILE%"
  echo [FATAL] Conda not found on PATH. Make sure Anaconda is installed and "conda init cmd.exe" was run.
  exit /b 1
)

REM Verify env activation by checking python path
echo --- Environment info --- >> "%LOGFILE%"
for /f "delims=" %%P in ('where python') do echo where python -> %%P>>"%LOGFILE%"

REM --- Paths ---
set PDFDIR=Textbooks\Chapter pdfs
set JSONDIR=Textbooks\Chapter json
set OUTDIR=data\gold_standard_extractions

echo --- Scanning "%PDFDIR%" for new chapters --- >> "%LOGFILE%"

REM --- Process only chapters without a gold_standard output ---
for %%f in ("%PDFDIR%\*.pdf") do (
  set "chapter=%%~nf"
  set "gold=%OUTDIR%\!chapter!_gold_standard.json"
  set "json=%JSONDIR%\!chapter!.json"

  if not exist "!gold!" (
    echo === Processing !chapter! ===
    echo === Processing !chapter! === >> "%LOGFILE%"

    if exist "!json!" (
      python tools\gold_standard_pipeline.py --single "%%f" --adobe-json "!json!" --model gpt-5 --verbose >> "%LOGFILE%" 2>&1
    ) else (
      python tools\gold_standard_pipeline.py --single "%%f" --model gpt-5 --verbose >> "%LOGFILE%" 2>&1
    )

    if errorlevel 1 (
      echo [ERROR] !chapter! failed. See log. 
      echo [ERROR] !chapter! failed. >> "%LOGFILE%"
    ) else (
      echo [OK] !chapter! completed.
      echo [OK] !chapter! completed. >> "%LOGFILE%"
    )
  ) else (
    echo Skipping !chapter! (already processed)
    echo Skipping !chapter! (already processed) >> "%LOGFILE%"
  )
)

echo ---------------------------------------------- >> "%LOGFILE%"
echo Done. Log saved to "%LOGFILE%".
echo Done. Log saved to "%LOGFILE%".
exit /b 0
