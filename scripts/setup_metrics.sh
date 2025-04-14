# clone common_metrics_on_video_quality repository
git clone git@github.com:CIntellifusion/common_metrics_on_video_quality.git 

# get IDM weights 
mkdir -p checkpoints/IDM
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model -O checkpoints/IDM/4x_idm.model
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights  -O checkpoints/IDM/4x_idm.weights
