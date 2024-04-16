$envFile = "PJA-ASI-12C-GR2.yaml"
$envName = "PJA-ASI-12C-GR2"

# Check if the file exists in the current path
if (-not (Test-Path $envFile)) {
    Write-Host "Error: '$envFile' not found." -ForegroundColor Red
    exit 1
}

# Verify if the specified environment already exists
$environments = conda env list
if ($environments -match $envName) {
    $decision = Read-Host "Environment with the same name already exists. Do you want to remove it? Type 'Yes' or 'No'"
    if ($decision -eq "Yes") {
        conda deactivate $envName
        conda activate
        conda remove -n $envName --all
        # Remove old folder (checks only default location)
        $defaultEnvPath = "C:\ProgramData\anaconda3\envs"
        $folderToRemove = Get-ChildItem -Path $defaultEnvPath | Where-Object { $_.Name -eq $envName }
        if ($folderToRemove.Name -eq $envName) {
            Remove-Item -Path $folderToRemove.FullName -Recurse -Force
        }
        Write-Host "Environment '$envName' successfully removed." -ForegroundColor Green
    }
    else {
        Write-Host "Cannot import new environment, exiting..." -ForegroundColor DarkYellow
        exit 1
    }
}

# Set strict priority channel
conda config --set channel_priority strict

# Import environment
conda env create -f $envFile -n $envName

# Activate environment
conda activate $envName
Write-Host "Environment '$envName' successfully created and activated." -ForegroundColor Green
conda env list