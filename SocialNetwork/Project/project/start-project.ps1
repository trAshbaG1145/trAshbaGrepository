param(
    [switch]$NoBrowser,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackDir = Join-Path $Root "JobRec_Back"
$FrontDir = Join-Path $Root "JobRec_Front"
$FlaskDir = Join-Path $FrontDir "Backend"
$LogsDir = Join-Path $Root "logs"

$SpringPort = 8090
$FlaskPort = 8081
$VitePort = 8089
$MySqlPort = 3413
$RedisPort = 6380
$Neo4jBoltPort = 7688

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Test-Port {
    param([int]$Port)
    $connection = Test-NetConnection -ComputerName "localhost" -Port $Port -WarningAction SilentlyContinue
    return [bool]$connection.TcpTestSucceeded
}

function Wait-Port {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port -Port $Port) {
            Write-Host "OK: $Name is listening on port $Port" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "$Name did not start listening on port $Port within $TimeoutSeconds seconds."
}

function Start-ServiceWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$Command
    )

    $escapedTitle = $Title.Replace("'", "''")
    $escapedWorkDir = $WorkingDirectory.Replace("'", "''")
    $escapedCommand = $Command.Replace("'", "''")
    $psCommand = "`$Host.UI.RawUI.WindowTitle = '$escapedTitle'; Set-Location '$escapedWorkDir'; $escapedCommand"

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $psCommand
    )
}

Write-Host "JobRec one-click startup" -ForegroundColor Yellow
Write-Host "Root: $Root"

Write-Step "Checking required commands"
Assert-Command "docker"
Assert-Command "npm.cmd"
Assert-Command "powershell.exe"

if (-not (Test-Path (Join-Path $BackDir "mvnw.cmd"))) {
    throw "Missing Maven wrapper: $BackDir\mvnw.cmd"
}

if (-not (Test-Path (Join-Path $FlaskDir "venv\Scripts\python.exe"))) {
    throw "Missing Python virtual environment: $FlaskDir\venv\Scripts\python.exe"
}

Write-Step "Starting Docker dependencies"
Push-Location $Root
try {
    docker compose up -d
}
finally {
    Pop-Location
}

Wait-Port -Port $MySqlPort -Name "Docker MySQL" -TimeoutSeconds 180
Wait-Port -Port $RedisPort -Name "Docker Redis" -TimeoutSeconds 120
Wait-Port -Port $Neo4jBoltPort -Name "Docker Neo4j Bolt" -TimeoutSeconds 180

Write-Step "Starting Spring Boot backend"
if (Test-Port -Port $SpringPort) {
    Write-Host "Spring Boot already appears to be running on port $SpringPort" -ForegroundColor Yellow
}
else {
    Start-ServiceWindow `
        -Title "JobRec Spring Boot :$SpringPort" `
        -WorkingDirectory $BackDir `
        -Command ".\mvnw.cmd spring-boot:run"
}

Write-Step "Starting Flask AI service"
if (Test-Port -Port $FlaskPort) {
    Write-Host "Flask already appears to be running on port $FlaskPort" -ForegroundColor Yellow
}
else {
    Start-ServiceWindow `
        -Title "JobRec Flask :$FlaskPort" `
        -WorkingDirectory $FlaskDir `
        -Command "`$env:FLASK_PORT='$FlaskPort'; .\venv\Scripts\python.exe app.py"
}

Write-Step "Starting Vue/Vite frontend"
if (Test-Port -Port $VitePort) {
    Write-Host "Vite already appears to be running on port $VitePort" -ForegroundColor Yellow
}
else {
    Start-ServiceWindow `
        -Title "JobRec Vite :$VitePort" `
        -WorkingDirectory $FrontDir `
        -Command "npm.cmd run dev -- --host 127.0.0.1"
}

Write-Step "Waiting for application ports"
Wait-Port -Port $SpringPort -Name "Spring Boot" -TimeoutSeconds 180
Wait-Port -Port $FlaskPort -Name "Flask" -TimeoutSeconds 120
Wait-Port -Port $VitePort -Name "Vite" -TimeoutSeconds 120

$FrontUrl = "http://localhost:$VitePort"
Write-Host ""
Write-Host "Project is running:" -ForegroundColor Green
Write-Host "  Frontend:    $FrontUrl"
Write-Host "  Spring Boot: http://localhost:$SpringPort"
Write-Host "  Flask AI:    http://localhost:$FlaskPort"
Write-Host "  Neo4j:       http://localhost:7475"

if (-not $NoBrowser) {
    Start-Process $FrontUrl
}

if (-not $NoPause) {
    Write-Host ""
    Read-Host "Press Enter to close this launcher window"
}
