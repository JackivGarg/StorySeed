$ErrorActionPreference = 'Stop'
$storyseedProject = Split-Path -Parent $PSScriptRoot
$storyseedPython = Join-Path $storyseedProject '.venv/Scripts/pythonw.exe'
if (-not (Test-Path -LiteralPath $storyseedPython)) {
    throw 'Project Python is missing. Follow docs/LOCAL_APP.md first.'
}
Start-Process -FilePath $storyseedPython -ArgumentList '-m', 'storyseed.app', '--gui' -WorkingDirectory $storyseedProject -WindowStyle Hidden
