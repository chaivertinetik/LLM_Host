#!/bin/bash
set -e  

# Micromamba
if ! command -v micromamba &>/dev/null; then
    echo "Micromamba not found. Installing..."
    
    # Create installation directory
    mkdir -p $HOME/micromamba
    
    # Download and install micromamba
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj -C $HOME/micromamba bin/micromamba
    else
        # Linux
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C $HOME/micromamba bin/micromamba
    fi
    
   
    export PATH="$HOME/micromamba/bin:$PATH"
    
    # Initialize micromamba 
    $HOME/micromamba/bin/micromamba shell init -s bash -p $HOME/micromamba/envs
    
  
    source $HOME/.bashrc
fi

# Check if environment exists, create if it doesn't
if ! micromamba env list | grep -q llm-geo; then
    echo "Creating llm-geo environment..."
    micromamba env create -f environment.yml
else
    echo "Updating llm-geo environment..."
    micromamba env update -f environment.yml
fi

# Activate environment and run the script
echo "Activating environment and running script..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate llm-geo

# Run the main script
echo "Running LLM_Geo_Executable.py..."
python LLM_Geo_Executable.py

# Deactivate environment
micromamba deactivate
echo "Process completed successfully!"