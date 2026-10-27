$ErrorActionPreference = 'Stop'
$sourcePath = 'C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image\build_ppt\font_normalized_candidate.pptx'
$resultPath = 'C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image\output\nuclear_morphology_presentation\Nuclear_morphology_study_plan_v14_regular.pptx'
$previewPath = 'C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image\build_ppt\previews_v14'
New-Item -ItemType Directory -Path $previewPath -Force | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open($sourcePath, -1, 0, 0)
    $count = 0
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $range = $shape.TextFrame.TextRange
                $range.Font.Name = 'Arial'
                $range.Font.NameFarEast = 'Arial'
                $range.Font.Bold = 0
                $shape.TextFrame2.TextRange.Font.Bold = 0
                if ($range.Font.Bold -ne 0) { throw 'Unable to clear bold' }
                $count++
            }
        }
    }
    $probe = $deck.Slides.Item(2).Shapes.Item(1).TextFrame.TextRange
    $probe.Font.Bold = -1
    if ($probe.Font.Bold -ne -1) { throw 'Bold-on check failed' }
    $probe.Font.Bold = 0
    if ($probe.Font.Bold -ne 0) { throw 'Bold-off check failed' }
    $deck.SaveAs($resultPath, 24)
    $deck.Export($previewPath, 'PNG', 1280, 720)
    $deck.Close()
    $deck = $app.Presentations.Open($resultPath, -1, 0, 0)
    $bad = 0
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                if ($shape.TextFrame.TextRange.Font.Bold -ne 0) { $bad++ }
            }
        }
    }
    if ($bad -ne 0) { throw "Still bold after reopening: $bad" }
    Write-Output "Normalized $count text objects. PowerPoint bold-on/off and save/reopen checks passed."
    Write-Output $resultPath
} finally {
    if ($null -ne $deck) { $deck.Close() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
