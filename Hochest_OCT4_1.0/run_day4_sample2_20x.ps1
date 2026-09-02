$pythonExe = "C:\Users\dodos\miniforge3\envs\cellpose\python.exe"
$scriptPath = Join-Path $PSScriptRoot "day2_trial.py"
$dataRoot = "E:\Kino-oka Lab\Immunostaining Data_Ekin\Immunostaining Data_Ekin\Day 4 Data\Sample 2"
$outputRoot = Join-Path $PSScriptRoot "outputs\day4_sample2_20x_trial"

& $pythonExe $scriptPath `
    --data-root $dataRoot `
    --output-root $outputRoot `
    --culture-day 4 `
    --sample "Sample 2" `
    --replicate "not_provided" `
    --fit-magnification "20x" `
    @args
exit $LASTEXITCODE
