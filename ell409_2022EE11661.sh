# #!/bin/bash

# # Script for ELL409 Project: Handwritten Digit Classification
# # Author: 2022EE11661
# # Date: $(date)

# # Step 1: Define environment name
# ENV_NAME="birds-env"

# # Step 2: Check if Conda is installed
# if ! command -v conda &> /dev/null
# then
#     echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
#     exit 1
# fi

# # Step 3: Check if environment.yml exists
# if [ ! -f "environment.yml" ]; then
#     echo "Error: environment.yml file not found!"
#     exit 1
# fi

# # Step 4: Create the environment from environment.yml
# echo "Creating conda environment '$ENV_NAME' from environment.yml..."
# conda env create -f environment.yml

# # Step 5: Activate the environment
# echo "Activating environment '$ENV_NAME'..."
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$ENV_NAME"

# # Step 6: Run install.sh (for pip-specific installations)
# if [ -f "install.sh" ]; then
#     echo "Running install.sh..."
#     bash install.sh
# else
#     echo "Warning: install.sh not found. Skipping pip installation step."
# fi

# # Step 7: Confirm environment setup
# echo "Environment '$ENV_NAME' is ready."
# echo "Python version: $(python --version)"
# echo "Installed packages:"
# pip list

# # Step 8: Log environment for reproducibility
# echo "Exporting environment to birds-env-lock.yml..."
# conda env export > birds-env-lock.yml

# echo "Setup complete!"

#!/bin/bash

# Set up paths
TRAIN_DATA="trainingdataset/mnist_1_4_8_train.npz"
VAL_DATA="recon-dataset/mnist_1_4_8_val_recon.npz"
MODEL_OUTPUT="vae.pth"
GMM_OUTPUT="gmm_params.pkl"

# Step 1: Create and activate virtual environment
echo "Creating virtual environment 'birds-env'..."
python3 -m venv birds-env
source birds-env/bin/activate

# Step 2: Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Run training script
echo "Running training..."
python setupfiles/vae.py "$TRAIN_DATA" "$VAL_DATA" train "$MODEL_OUTPUT" "$GMM_OUTPUT"

echo "Training completed!"
