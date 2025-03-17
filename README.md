# LLM_Host
This repository contains the hosted Github CI/CD for LLM Erdo


The repo offers two functionalities :
1) Running the LLM Assitant script locally: installs micromamba directly from micro.mamba.pm and includes OS detection to download the correct version for macOS or Linux

and 

2) For CI/CD automated workflow: first creates a virutal ubuntu machine and then clones the code to the runner. Then creates the virtual environment using the environment file. It finally activate the llm-geo environment, installs any dependencies from requirements.txt and executes your LLM_Geo_Executable.py script.


The worflow automation triggers when: Someone opens a pull request against the main branch or You manually trigger it via the GitHub Actions UI using the "workflow_dispatch" event
