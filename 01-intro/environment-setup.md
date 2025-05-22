# Environment Setup

## Create AWS EC2 Instance

1. Launch instance
2. Name it mlops-zoomcamp
3. Select ubuntu AMI (Ubuntu Server 22.04 LTS (HVM) SSD Volume Type)
4. Select archtecture 64-bit (x86)
5. Select instance type t2.xlarge (4 vCPU 16 GiB Memory)
6. Create key pair RSA in .pem format and place key in your .ssh folder
7. Select 30 GiB for configure storage
8. Launch instance

* Reminder to stop instance when not working on it.
* IP address changes every time instance is stopped

To access instance

```ssh -i ~/.ssh/<key_pair_name> ubuntu@<public_IPv4_address>```

To automatically connect to the instance, update ~/.ssh/config file

```
Host mlops-zoomcamp
    HostName <public_IPv4_address>
    User ubuntu
    IdentityFile ~/.ssh/<key_pair_name>
    StrictHostKeyChecking no
```

Remember to update the file with the new public IPv4 address, if you have stopped your instance

Now you can use this command to access the instance
```ssh mlops-zoomcamp```

### Setup Budget

1. Go to Billing and Cost Management on your AWS root user account.
2. Select Budgets
3. Create a budget
4. Select monthly cost budget
5. Write in budget name
6. Write in budgeted amount in $0.00 format
7. Specify email recipients when threshold has exceeded
8. Create budget

## Check Python

Run this command in the instance to check the version of python

```python3```

## Install Anaconda

Install anaconda for linux 64-bit (x86)

```wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh```

Execute

```bash Anaconda3-2024.10-1-Linux-x86_64.sh```

Follow instructions to install Anaconda

Check python

```which python```

Output

```/home/ubuntu/anaconda3/bin/python```

Check python version

```python```

Output 

```Python 3.12```

## Install Docker

```
sudo apt update
sudo apt install docker.io
```

Download docker-compose and make it executable

```
mkdir soft
cd soft
wget https://github.com/docker/compose/releases/download/v2.36.0/docker-compose-darwin-x86_64 -O docker-compose
ls -l
chmod +x docker-compose
```

Access soft folder from any location by updating bash file

```vim .bashrc```

Add to end of file

```export PATH = "${HOME}/soft:${PATH}"```

Re-execute file

```
source .bashrc
which docker-compose
```

Output

```/home/ubuntu/soft/docker-compose```

Check if Docker runs

```docker run hello-world```

If it doesn't work without sudo, do the following

```
sudo groupadd docker
sudo usermod -aG docker $USER
```

Logout and login

Try again

```docker run hello-world```

## Connect Public Git Repo to AWS EC2 Instance

Source: 09 clone private repo in AWS EC2. Lets Write Code. https://youtu.be/cEyn7RHu7-c?si=zirI4SlCSN9-mlbg

Check if git is installed

```sudo apt install git```

Create ssh key pair

```ssh-keygen```

Get the public key

```cat ~/.ssh/<key_name>.pub```

Copy the public key. Go to Github Settings, SSH and GPG Keys to add key.

Authenticate connection to github

```ssh -T git@github.com```

```git clone <git_repo_ssh_url>```

To make sure you can also git push, check the following:

```git remote -v```

If output is this:
```
origin  https://github.com/<USERNAME>/<REPONAME>.git (fetch)
origin  https://github.com/<USERNAME>/<REPONAME>.git (push)
```

Switch to SSH
```git remote set-url origin git@github.com:<USERNAME>/<REPONAME>.git```

Output:
```
origin git@github.com:<USERNAME>/<REPONAME>.git (fetch)
origin git@github.com:<USERNAME>/<REPONAME>.git (push)
```

## Access Visual Studio Code

Connect remotely to EC2 instance

* Remote - SSH extension
* Connect to host mlops-zoomcamp

## Start Jupyter Notebook

In EC2 instance make a new directory notebooks

```
mkdir notebooks
cd notebooks
jupyter notebook
```

In VSCode, use port forwarding

* Open up terminal, select Ports
* Forward a port
* Port 8888
* Copy url for jupyter notebook to local browser
