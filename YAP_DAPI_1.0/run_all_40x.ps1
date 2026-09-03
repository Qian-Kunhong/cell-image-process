$pythonExe = "C:\Users\dodos\miniforge3\envs\cellpose\python.exe"
$scriptPath = Join-Path $PSScriptRoot "yap_40x_trial.py"
$dataRoot = "E:\Kino-oka Lab\Immunostaining Data_Ekin\2307YapLocalizationImmuno"
$outputRoot = Join-Path $PSScriptRoot "outputs\all_40x_trial"

& $pythonExe $scriptPath `
    --data-root $dataRoot `
    --output-root $outputRoot `
    --fit-magnification "40x" `
    @args
exit $LASTEXITCODE
