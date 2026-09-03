$pythonExe = "C:\Users\dodos\miniforge3\envs\cellpose\python.exe"
$scriptPath = Join-Path $PSScriptRoot "day2_trial.py"

& $pythonExe $scriptPath @args
exit $LASTEXITCODE
