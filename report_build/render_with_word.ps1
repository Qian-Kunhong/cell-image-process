$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskDoc = Join-Path $taskRoot 'reports\Stem_Cell_Nuclear_Phenotyping_Research_Plan_Bilingual.docx'
$taskPdf = Join-Path $taskRoot 'report_build\render\preview.pdf'
New-Item -ItemType Directory -Force (Split-Path $taskPdf) | Out-Null
Write-Output 'Starting Word renderer'
Start-Process -FilePath 'C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE' -ArgumentList '/a /q' -WindowStyle Hidden
Start-Sleep -Seconds 3
$taskWord = New-Object -ComObject Word.Application
Write-Output 'Word ready'
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
$taskWord.AutomationSecurity = 3
try {
    Write-Output ('Opening ' + $taskDoc)
    $taskDocument = $taskWord.Documents.OpenNoRepairDialog($taskDoc, $false, $true)
    Write-Output 'Document opened'
    $taskWord.ActivePrinter = 'Microsoft Print to PDF'
    Write-Output 'Exporting preview'
    $taskDocument.ExportAsFixedFormat($taskPdf, 17)
    Write-Output 'Preview exported'
    $taskDocument.Close(0)
} finally {
    $taskWord.Quit()
}
