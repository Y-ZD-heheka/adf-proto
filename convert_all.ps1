# Batch ADF round-trip for every STL in .\stls  ->  .\out\<name>.stl + .\out\<name>.log
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File .\convert_all.ps1
# Optional: -HTarget 0.1  -Only "flower" (substring filter)  -MaxDepth 11 (default: auto = 11 only for large thin-walled parts, else 10)  -Tol 0.02 (max deviation mm; 0.01 = finest, larger = smaller file)
param([double]$HTarget = 0.1, [string]$Only = "", [int]$MaxDepth = 0, [double]$Tol = 0.02)
if ($MaxDepth -gt 0) { $env:ADF_MAX_DEPTH = "$MaxDepth" } else { Remove-Item Env:ADF_MAX_DEPTH -ErrorAction SilentlyContinue }
$depthLabel = if ($MaxDepth -gt 0) { "$MaxDepth" } else { "auto" }
$env:ADF_REF_TOL = "$Tol"
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot
cargo build --release --bin stl_roundtrip
if ($LASTEXITCODE -ne 0) { Write-Host "build failed"; exit 1 }
New-Item -ItemType Directory -Force out | Out-Null
$exe = ".\target\release\stl_roundtrip.exe"
$summary = @()
Get-ChildItem stls -File | Where-Object { $_.Extension -ieq ".stl" -and $_.Name -like "*$Only*" } | ForEach-Object {
    $name = $_.BaseName
    $out  = Join-Path "out" ($name + ".stl")
    $log  = Join-Path "out" ($name + ".log")
    Write-Host "==== $($_.Name)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $exe $_.FullName $HTarget $out 1 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $sw.Stop()
    $size = if (Test-Path $out) { [math]::Round((Get-Item $out).Length / 1MB, 2) } else { -1 }
    $line = "{0}`th={1} tol={6} depth={7}`texit={2}`t{3:n1}s`tin={4:n2}MB`tout={5}MB" -f $name, $HTarget, $code, $sw.Elapsed.TotalSeconds, ($_.Length / 1MB), $size, $Tol, $depthLabel
    Add-Content -Path $log -Value "[batch] $line"
    $summary += $line
}
$summary | Set-Content (Join-Path "out" "_summary.txt")
Write-Host ""
Write-Host "==== summary"
$summary | ForEach-Object { Write-Host $_ }
