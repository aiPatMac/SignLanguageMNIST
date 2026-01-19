# Automated Deployment for Sign Language MNIST
if (Test-Path ".env") {
    Get-Content .env | Where-Object { $_ -match '=' -and -not $_.StartsWith('#') } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        Set-Variable -Name $name.Trim() -Value $value.Trim() -Scope Script
    }
} else {
    Write-Error ".env file not found! Please create one based on the template."
    exit
}

Write-Host "--- Starting Deployment to $EC2_IP ---" -ForegroundColor Cyan

Write-Host "[1/5] Preparing remote directories..." -ForegroundColor Yellow
ssh -i $KEY_PATH ec2-user@$EC2_IP "mkdir -p /home/ec2-user/models"

Write-Host "[2/5] Uploading source code and model (this may take a minute)..." -ForegroundColor Yellow
scp -i $KEY_PATH requirements.txt Dockerfile ec2-user@${EC2_IP}:/home/ec2-user/
scp -i $KEY_PATH -r src ec2-user@${EC2_IP}:/home/ec2-user/
scp -i $KEY_PATH $LOCAL_CHECKPOINT ec2-user@${EC2_IP}:/home/ec2-user/models/toy_model.ckpt

Write-Host "[3/5] Setting up EC2 Environment (Docker, Swap, Build)..." -ForegroundColor Yellow
$RemoteCommand = @"
    # Setup Swap File (Emergency RAM)
    if [ ! -f /swapfile ]; then
        sudo dd if=/dev/zero of=/swapfile bs=128M count=16
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi

    # Install/Start Docker
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user

    # Build the Docker Image
    cd /home/ec2-user
    sudo docker build -t sign-mnist .
"@

ssh -i $KEY_PATH ec2-user@$EC2_IP $RemoteCommand

Write-Host "[4/5] Starting the Container..." -ForegroundColor Yellow
$RunCommand = @"
    sudo docker rm -f sign-mnist-prod || true
    sudo docker run -d -p 8000:8000 --name sign-mnist-prod sign-mnist
"@
ssh -i $KEY_PATH ec2-user@$EC2_IP $RunCommand

Write-Host "[5/5] Waiting for startup and verifying..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "--- Local Test Request ---" -ForegroundColor Cyan
python -m src.service.send_example --index 5 --url "http://$($EC2_IP):8000/predict"

Write-Host "--- Deployment Complete ---" -ForegroundColor Green