param(
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$runtimeRoot = Join-Path $projectRoot ".dev-runtime"
$backendLog = Join-Path $runtimeRoot "backend.log"
$backendErrorLog = Join-Path $runtimeRoot "backend-error.log"
$frontendLog = Join-Path $runtimeRoot "frontend.log"
$frontendErrorLog = Join-Path $runtimeRoot "frontend-error.log"
$backendUrl = "http://127.0.0.1:8000/health"
$frontendUrl = "http://127.0.0.1:5173"

function Test-PortInUse {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync("127.0.0.1", $Port)
        return $connection.Wait(300) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-ServiceReady {
    param(
        [string]$Name,
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 60
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "$Name 启动进程已退出，退出码：$($Process.ExitCode)"
        }

        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    throw "等待 $Name 启动超时：$Url"
}

function Show-LogTail {
    param(
        [string]$Title,
        [string]$Path
    )

    if (Test-Path -LiteralPath $Path) {
        Write-Host "`n[$Title] $Path" -ForegroundColor Yellow
        Get-Content -LiteralPath $Path -Tail 20 -Encoding UTF8
    }
}

$backendProcess = $null
$frontendProcess = $null

try {
    if (Test-PortInUse -Port 8000) {
        throw "端口 8000 已被占用。请先关闭现有后端进程。"
    }
    if (Test-PortInUse -Port 5173) {
        throw "端口 5173 已被占用。请先关闭现有前端进程。"
    }

    $pythonCommand = Get-Command python -ErrorAction Stop
    $nodeCommand = Get-Command node -ErrorAction Stop
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) {
        $npmCommand = Get-Command npm -ErrorAction Stop
    }

    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

    & $pythonCommand.Source -c "import fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "首次运行：正在安装 Python 依赖..." -ForegroundColor Cyan
        & $pythonCommand.Source -m pip install -r (Join-Path $projectRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Python 依赖安装失败。"
        }
    }

    $viteEntry = Join-Path $frontendRoot "node_modules\vite\bin\vite.js"
    if (-not (Test-Path -LiteralPath $viteEntry)) {
        Write-Host "首次运行：正在安装前端依赖..." -ForegroundColor Cyan
        Push-Location $frontendRoot
        try {
            & $npmCommand.Source ci
            if ($LASTEXITCODE -ne 0) {
                throw "前端依赖安装失败。"
            }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "正在启动后端..." -ForegroundColor Cyan
    $backendProcess = Start-Process `
        -FilePath $pythonCommand.Source `
        -ArgumentList @("-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $backendLog `
        -RedirectStandardError $backendErrorLog `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "正在启动前端..." -ForegroundColor Cyan
    $frontendProcess = Start-Process `
        -FilePath $nodeCommand.Source `
        -ArgumentList @($viteEntry, "--host", "127.0.0.1", "--port", "5173") `
        -WorkingDirectory $frontendRoot `
        -RedirectStandardOutput $frontendLog `
        -RedirectStandardError $frontendErrorLog `
        -WindowStyle Hidden `
        -PassThru

    Wait-ServiceReady -Name "后端" -Url $backendUrl -Process $backendProcess
    Write-Host "后端已就绪：$backendUrl" -ForegroundColor Green

    Wait-ServiceReady -Name "前端" -Url $frontendUrl -Process $frontendProcess
    Write-Host "前端已就绪：$frontendUrl" -ForegroundColor Green

    [pscustomobject]@{
        backend_pid = $backendProcess.Id
        frontend_pid = $frontendProcess.Id
        started_at = [DateTime]::Now.ToString("s")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeRoot "processes.json") -Encoding UTF8

    if (-not $NoBrowser) {
        Start-Process $frontendUrl
    }

    Write-Host "`n所有服务已启动。浏览器地址：$frontendUrl" -ForegroundColor Green
    Write-Host "按 Ctrl+C 可同时关闭前端和后端。" -ForegroundColor Yellow
    Write-Host "日志目录：$runtimeRoot`n"

    while (-not $backendProcess.HasExited -and -not $frontendProcess.HasExited) {
        Start-Sleep -Seconds 1
    }

    if ($backendProcess.HasExited) {
        throw "后端意外退出，退出码：$($backendProcess.ExitCode)"
    }
    throw "前端意外退出，退出码：$($frontendProcess.ExitCode)"
}
catch {
    Write-Host "`n启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Show-LogTail -Title "后端错误日志" -Path $backendErrorLog
    Show-LogTail -Title "前端错误日志" -Path $frontendErrorLog
    exit 1
}
finally {
    foreach ($process in @($frontendProcess, $backendProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "前端和后端已关闭。" -ForegroundColor DarkGray
}
