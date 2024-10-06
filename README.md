# Research

## Import Modules

```powershell
if (-not (Get-Module -ListAvailable -Name posh-git)) {
    Install-Module posh-git -Scope CurrentUser
}
Import-Module posh-git

if (-not (Get-Module -ListAvailable -Name oh-my-posh)) {
    Install-Module oh-my-posh -Scope CurrentUser
}
Import-Module oh-my-posh

# Initialize oh-my-posh
oh-my-posh init pwsh --config "C:\Users\syrym\AppData\Local\oh-my-posh\paradox.omp.json" | 
Invoke-Expression
```

```powershell
# Set Git Aliases in lowercase
Set-Alias gstatus git
Set-Alias gcommit git
Set-Alias gpush git
Set-Alias gadd git
```

```powershell
# Start SSH Agent if not running
$sshAgentStatus = Get-Process ssh-agent -ErrorAction SilentlyContinue
if (-not $sshAgentStatus) {
    Start-Service ssh-agent
    Set-Service -Name ssh-agent -StartupType Automatic
}
```

```powershell
# Add SSH key if not already added
$sshKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
if (Test-Path $sshKeyPath) {
    $sshKeys = ssh-add -l 2>&1
    if ($sshKeys -match "The agent has no identities") {
        ssh-add $sshKeyPath
        Write-Host "SSH key added: $sshKeyPath"
    } else {
        Write-Host "SSH key already added."
    }
}
```

```powershell
# Add Scripts Directory to PATH
$scriptsPath = "C:\Users\syrym\scripts"
if (-not ($env:PATH -split ";" | Where-Object { $_ -ieq $scriptsPath })) {
    $env:PATH += ";$scriptsPath"
    [System.Environment]::SetEnvironmentVariable("PATH", $env:PATH, [System.EnvironmentVariableTarget]::User)
    Write-Host "Added scripts directory to PATH: $scriptsPath"
}
```

```powershell
# GitHub CLI Automation Functions

function New-Repo {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [string]$Description = "",

        [switch]$Private
    )
    $visibility = if ($Private) { "--private" } else { "--public" }
    gh repo create $Name --description "$Description" $visibility --confirm
}
Set-Alias nr New-Repo

function Get-RepoHttp {
    gh repo list --limit 100 --json url | ConvertFrom-Json | Select-Object -ExpandProperty url
}
Set-Alias lrhttp Get-RepoHttp

function Get-RepoSsh {
    gh repo list --limit 100 --json sshUrl | ConvertFrom-Json | Select-Object -ExpandProperty sshUrl
}
Set-Alias lrssh Get-RepoSsh

function Remove-RepoSsh {
    param(
        [Parameter(Mandatory=$true)]
        [string]$RepoFullName
    )
    gh repo delete $RepoFullName --ssh --confirm
}
Set-Alias rrssh Remove-RepoSsh

function Remove-RepoHttp {
    param(
        [Parameter(Mandatory=$true)]
        [string]$RepoFullName
    )
    gh repo delete $RepoFullName --confirm
}
Set-Alias rrhttp Remove-RepoHttp

function Update-Profile {
    try {
        . $PROFILE
        Write-Host "PowerShell profile reloaded successfully."
    }
    catch {
        Write-Host "Failed to reload PowerShell profile."
    }
}
Set-Alias updatep Update-Profile

function Edit-Profile {
    notepad $PROFILE
}
Set-Alias ep Edit-Profile
```

```powershell
# Enable Tab auto-completion
Set-PSReadLineKeyHandler -Key Tab -Function Complete

Set-Alias conda "C:\Users\syrym\miniconda3\Scripts\conda.exe"
```

```powershell
# Manually initialize Conda
& "C:\Users\syrym\miniconda3\shell\condabin\conda-hook.ps1" | Out-Null
```
