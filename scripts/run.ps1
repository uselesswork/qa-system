param(
    [ValidateSet('start', 'status', 'stop')]
    [string]$Action = 'start'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runtimeDir = Join-Path $projectRoot '.runtime'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$composeFile = Join-Path $projectRoot 'compose.ingestion.yaml'
$services = @('worker', 'beat', 'web')
Set-Location -LiteralPath $projectRoot

function Get-ManagedProcess([string]$name) {
    $stateFile = Join-Path $runtimeDir "$name.json"
    if (-not (Test-Path -LiteralPath $stateFile)) { return $null }
    try {
        $record = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$record.pid) -ErrorAction Stop
        if ($process.StartTime.ToUniversalTime().Ticks -ne [long]$record.startTicks) { return $null }
        return $process
    } catch { return $null }
}

function Start-ManagedProcess([string]$name, [string[]]$processArgs) {
    $existing = Get-ManagedProcess $name
    if ($null -ne $existing) {
        Write-Host "$name 已运行（PID $($existing.Id)）。"
        return
    }
    $stateFile = Join-Path $runtimeDir "$name.json"
    if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
    $stdoutFile = Join-Path $runtimeDir "$name.out.log"
    $stderrFile = Join-Path $runtimeDir "$name.err.log"
    $process = Start-Process -FilePath $pythonExe -ArgumentList $processArgs `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    Start-Sleep -Milliseconds 400
    $process.Refresh()
    if ($process.HasExited) {
        throw "$name 启动后立即退出。请查看 $stderrFile"
    }
    @{ pid = $process.Id; startTicks = $process.StartTime.ToUniversalTime().Ticks } |
        ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8
    Write-Host "$name 已启动（PID $($process.Id)）；日志：$stderrFile"
}

function Test-WebPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', 8090, $null, $null)
        return ($pending.AsyncWaitHandle.WaitOne(300) -and $client.Connected)
    } catch { return $false }
    finally { $client.Close() }
}

switch ($Action) {
    'start' {
        if (-not (Test-Path -LiteralPath $pythonExe)) { throw "缺少虚拟环境：$pythonExe" }
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw '缺少 .env 配置文件。' }
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw '找不到 docker 命令。请安装并启动 Docker Desktop。' }
        docker info --format '{{.ServerVersion}}' *> $null
        if ($LASTEXITCODE -ne 0) { throw 'Docker 引擎尚未就绪。请先启动 Docker Desktop。' }
        if ((Test-WebPort) -and $null -eq (Get-ManagedProcess 'web')) {
            throw '8090 端口已有其他进程在使用。请先关闭原来手动启动的 Web 终端。'
        }
        docker compose -f $composeFile up -d
        if ($LASTEXITCODE -ne 0) { throw 'Redis 或 Qdrant 容器启动失败。' }
        New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
        Start-ManagedProcess 'worker' @('-m', 'celery', '-A', 'src.tasks.ingest:celery_app', 'worker', '--pool=solo', '--loglevel=info')
        Start-ManagedProcess 'beat' @('-m', 'celery', '-A', 'src.tasks.ingest:celery_app', 'beat', '--loglevel=info', '--schedule', '.runtime/celerybeat-schedule')
        Start-ManagedProcess 'web' @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8090')
        Write-Host '启动命令已提交。打开 http://127.0.0.1:8090/；用 .\run.cmd status 查看状态。'
    }
    'status' {
        foreach ($name in $services) {
            $process = Get-ManagedProcess $name
            if ($null -ne $process) { Write-Host "$name : 运行中（PID $($process.Id)）" }
            else { Write-Host "$name : 未由脚本运行" }
        }
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            docker compose -f $composeFile ps
        }
    }
    'stop' {
        foreach ($name in @('web', 'beat', 'worker')) {
            $process = Get-ManagedProcess $name
            if ($null -ne $process) {
                Stop-Process -Id $process.Id
                Write-Host "$name 已停止。"
            }
            $stateFile = Join-Path $runtimeDir "$name.json"
            if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }
        }
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            docker compose -f $composeFile stop
            if ($LASTEXITCODE -ne 0) { throw '容器停止失败。' }
        }
    }
}
