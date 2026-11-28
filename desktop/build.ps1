$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m PyInstaller --noconfirm --clean --windowed --onedir --name AgentMilieu --paths src/agentmilieu --specpath build desktop/agentmilieu_desktop.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$compiler = "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
if (-not (Test-Path $compiler)) { throw "Install NSIS 3 before building the installer." }
$version = python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])'
if ($LASTEXITCODE -ne 0) { throw "Could not read the package version." }
New-Item -ItemType Directory -Force "dist/windows" | Out-Null
& $compiler "/DAPP_VERSION=$version" "desktop/AgentMilieu.nsi"
if ($LASTEXITCODE -ne 0) { throw "NSIS failed." }

$smokeDir = Join-Path ([System.IO.Path]::GetTempPath()) "AgentMilieu-Smoke-$([guid]::NewGuid())"
try {
    $installer = Start-Process -FilePath "dist/windows/AgentMilieu-Setup.exe" -ArgumentList "/S /D=$smokeDir" -Wait -PassThru
    if ($installer.ExitCode -ne 0) { throw "The Windows installer failed." }
    $app = Start-Process -FilePath "$smokeDir/AgentMilieu.exe" -ArgumentList "--smoke-test" -Wait -PassThru
    if ($app.ExitCode -ne 0) { throw "The installed Windows application did not start." }
}
finally {
    if (Test-Path "$smokeDir/Uninstall.exe") {
        Start-Process -FilePath "$smokeDir/Uninstall.exe" -ArgumentList "/S" -Wait
    }
}
