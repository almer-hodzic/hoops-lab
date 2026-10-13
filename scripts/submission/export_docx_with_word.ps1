param(
    [Parameter(Mandatory=$true)][string]$DocxPath,
    [Parameter(Mandatory=$true)][string]$PdfPath,
    [Parameter(Mandatory=$true)][string]$InfoPath
)

$ErrorActionPreference = "Stop"
$word = $null
$doc = $null
try {
    $resolvedDocx = (Resolve-Path -LiteralPath $DocxPath).Path
    $resolvedPdf = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PdfPath)
    $resolvedInfo = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InfoPath)
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($resolvedDocx, $false, $false)
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
    $pageCount = $doc.ComputeStatistics(2)
    $doc.ExportAsFixedFormat($resolvedPdf, 17)
    @{
        docx = $resolvedDocx
        pdf = $resolvedPdf
        page_count = $pageCount
    } | ConvertTo-Json | Set-Content -LiteralPath $resolvedInfo -Encoding UTF8
}
finally {
    if ($doc -ne $null) { $doc.Close($false) | Out-Null }
    if ($word -ne $null) { $word.Quit() | Out-Null }
}
