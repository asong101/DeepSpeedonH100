# DeepSpeedonH100
https://dl.acm.org/doi/10.1145/3634769.3634806
https://github.com/microsoft/DeepSpeed
 
Learn about how the models actually work
https://arxiv.org/abs/1706.03762
 
 
Installed pytorch, use on desktop with Cuda GPU? use locally on Conda or on Colab
Use colab to learn, then run locally with a cloud gpu
https://pytorch.org/get-started/locally/#windows-verification
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
https://github.com/pytorch/pytorch#install-pytorch
 
Test Runs
-try different platforms, cloud gpus
-think about what models to optimize
-download needed apps on desktop, try to run some programs
Conda
Pytorch Cuda
Transformers
 
-utilization of the GPU, waste power if not high percent utilization
-loading data
-batch size
Learn how to optimize different things
Wandb
https://wandb.ai/site
https://github.com/wandb/examples
Try with colab, Lambda (Jupyter), etc.
 
Linux for Cloud
-no man command on git bash
Lecture 3: Editors (vim) (2020)
 

 

 
5/18
 
Fix Lambda issues
Once fixed, try digit classifier using different GPUs and update Prof
Try Colab, Lambda, other cloud GPUs
 
 
5/22
Learn Deepspeed
 
Try lambda to stop syncing or try different cloud gpu
Test gpu speed, # steps, learning rate
 
5/26
Try codecarbon
Try lambda/wandb using jupyter
 
Call lambda, fix wandb syncing
 
5/30
Try codecarbon



https://huggingface.co/EleutherAI/gpt-neo-125m 
GPUs are faster for matrix processing and number crunching, just like in graphics
https://docs.google.com/presentation/d/13F11zqAzQTqna7k0_BrhBhHzLww39ptObDrUgsr3n-s/edit#slide=id.p 
https://inspiritaischolars.teachable.com/courses/enrolled/2017222 

Train vs inference - train back-propagates parameters to become more accurate, inference just uses trained model on a dataset

Foundation model - trained to do different things, adapted to a wide range of downstream tasks,
Fine-tuning: using a foundation model fine tuning it to learn a new skill, much less expensive than simply training the foundation
https://blogs.nvidia.com/blog/2023/03/13/what-are-foundation-models/#:~:text=Foundation%20models%20are%20AI%20neural,text%20to%20analyzing%20medical%20images 



https://hai.stanford.edu/  https://crfm.stanford.edu/research.html 
Human-centered - worried about AI dominance, gpt all uses human centered

Chat-gpt - iphone moment for AI, showed everyday people how the technology can be used

Diffusion models - art instead of text

https://app.leonardo.ai/ 
https://imagen.research.google/ 
https://stablediffusionweb.com/ 

Batch vs Epoch in in neural network

Learning rate
gradient decent - gradually getting closer and closer to optimal min
perplexity - how well probability model predicts a sample

large data set - larger learning rate to accommodate for more, small data set don’t change as much so use smaller learning rate













Research Goal
First need to understand carbon footprint and how bad it is
Then learn how to deploy model on gpu

Carbon impacts can and should be mitigated in many cases. This can be accomplished by training models in low-carbon intensity regions, or by using more efficient models and hardware (§5.3.1: environment-mitigation).
-day vs night, fluctuations in amount of users (use green energy unless more users), build facility in low-carbon regions

 (2) When all mechanisms for mitigation have been exhausted and mitigation is no longer possible, the costs and benefits to society should be assessed to determine if and when a larger foundation model should be deployed over a smaller, more efficient, model — with the understanding that the up-front costs of a large foundation model may be amortized over the lifetime of the model (§5.3.2: environment-costs). 
(3) Energy, computational, and carbon costs — as well as any efforts taken to mitigate negative impacts — should be clearly reported to inform policymaking and research (§5.3.3: environment reporting). 

This will be achieved as the model is being optimized
























Foundation Models Risk Research Paper https://arxiv.org/pdf/2108.07258.pdf 

Effective on many tasks → incentivizes uniformity, demands caution in making the foundation models, defects inherited by adapted models,


Research on Fmodels require interdisciplinary collaboration with respect to their interrelatedness with society


New trend on building AI on foundation models, trained on broad data adapted to any tasks [ex BERT, GPT3, CLIP]
Not novel, based on deep NN and self-supervised learning, newer ones are much more complex, GPT3 has 175bil parameters for a NLP foundation model


We lack understanding of their failures and potential to harm, poorly understood, increasing deployment, unanticipated consequences


Homogenization - consolidation, many possible tasks but uniform points of failure  
Deep learning - high-level prediction, Fmodel - in-context learning
Machine learning homogenize - learning algorithms(logistic regression), deep learning homogenize model architectures (CNN), foundation models homogenize - model itself
ML - wide range of applications powered by single learning algo - log reg

Machine learning - prediction trained on data to make future  predictions, instead of how to solve a task, induce it based on previous data (supervised then unsupervised)
NLP & comp vision such as answer questions or object recognition require “feature engineering” - write logic to convert data to features more suitable for ML

Deep learning - > dataset, more computation (better GPU) deep NN trained on raw inputs (pixels) more complex features through training, shift to homogenization same NN for many applications

Fmodel - most strongly in NLP, general trend in AI just like deep learning popular in computer vision but used otherwise
Transfer learning (makes Foundation possible)- learned knowledge of one task to another 
Pretraining best approach (trained on substitute task (means to an end), adapted via fine tuning to other tasks
Scale (makes Foundation powerful) - GPU throughput/memory increase, development of Transformer model architecture use parallelism of hardware train expressive models, more training data

Transfer learning w annotated datasets common (pretraining on ImageNet dataset for image classification) cost of annotation limit on pretraining
Self-supervised learning - pretraining derived from unannotated, more scalable (depend on unlabeled data, force model to predict, more useful than models trained on limited labelled space)

Before NLP associate word with context-independent vector, but now predict next word given previous words (in context)
New self-sup learning development embrace Transformer architecture, more powerful encoding for sentence, scale higher, used across research communities
Multimodal models - trained on lang and vision

Social impact - People feed data into the model, ultimately receive benefits and harms, must focus on the deployment of a model trained with questionable data
Need for adaptation across different tasks
Research only done in-industry, need disciplinary diversity
Loss in accessibility  for foundation models
Intense experimentation is needed
Startups find difficult to compete, centralizing nature of foundation models raises needs resources for developing them

Capabilities - process diff modalities, affect physical world, reasoning, interact with humans NLP - foundation, vision - deep learning, robotics
Biomed, law, education…

Tech behind it - modeling - expressivity (assimilate real-world info) scalability (large quants)

Low carbon-intensity regions
	














Machine Learning Models

Testing with GPT models easier because you can objectively tell what is better, better for publishing research and tell if optimization is viable
 
Use code to get IP of where code is being run, can figure out carbon impact
 
MNIST digit classifier
https://nextjournal.com/gkoehler/pytorch-mnist

Research Questions

Why does this matter?
What is your idea to reduce? ( show methodology)
-reduce training (when to stop training,
Diminishing return, 


how much accuracy it improves every train, margin of how accurate it is for specific task, test different GPUs
-learning rate - speed of neural network (advance quickly or slowly), how aggressive 
neural network changes parameters (if speed too high, change too quickly based off too little information)
https://stackoverflow.com/questions/59737875/keras-change-learning-rate 

What is the effectiveness of your newly established model? Prove through quality of results
Experiments










Lambda/Cloud GPU ubuntu server

https://portal.azure.com/?quickstart=true#home 
https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
https://www.youtube.com/watch?v=tH3fhhQB0vo&ab_channel=TutLinks 

https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines
https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu 
https://medium.com/mcgill-artificial-intelligence-review/tutorial-setting-up-a-gpu-enabled-virtual-machine-on-microsoft-azure-f9a32fa1b536

find path of something
find / -type d -name "cuda"

Install Conda
https://lambdalabs.com/blog/setting-up-a-anaconda-environment 
https://repo.anaconda.com/archive/ 

wget [version right click for get address]
https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
-https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

sh [file that was downloaded]
-Anaconda3-2022.10-Linux-x86_64.sh

Close and reopen current shell




Wandb

Wandb run on Lambda
https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization 
cloud gpu 
https://cloud.google.com/gpu  
https://gitforwindows.org/ terminal to run on cloud








https://community.wandb.ai/t/integration-of-wandb-with-aws-lambda/2280 
Error: __init__() got an unexpected keyword argument 'no_args_is_help'

SOLUTION: pip3 install click --upgrade


New issue with wandb.init()


Solution: pip install protobuf==3.19.0





-Lambda Update everything
sudo apt-get update && sudo apt-get dist-upgrade

-Update python
https://cloudbytes.dev/snippets/upgrade-python-to-latest-version-on-ubuntu-linux
https://linuxhint.com/update-python-ubuntu/ 

-Activate SSH
ssh -i [name].pem ubuntu@[IP]

-Reboot system
Sudo reboot

-See permissions of directory
Ls -l




























Run MNIST on Colab


https://github.com/pytorch/examples/issues/369 
Change args = parser.parse_args() to this line args = parser.parse_args(args=[])




Decrease Cost with Deepspeed
https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-chat/README.md 

Versatile GPT models more accessible, but lack of 
RLHF (Reinforced Learning from Human Feedback)
To train powerful GPT model that is also accessible

Deepspeed-Chat:
-easy training for GPT model, run pretrained huggingface through InstructGPT training using DeepSpeed-RLHF and produce own GPT model
-training pipeline replaicates Instruct GPT includes Supervised Fine-tuning(SFT), Reward Model Fine-tuning, RLHF, data abstraction/blending to enable train multiple data source
-DeepSpeed-RLHF combines trianing/inference of DS to Hybrid Engine, transition b/w inference & training, leverage optimizations (tensor-parallelism, performance transformer kernels for generation), ZeRO/LoRA memory optimization strat for Rein Learning training, <DS makes optimal memory mgmt & data mvmt across phases of RLHF

Install Conda


export PATH=”/home/ubuntu/anaconda3/bin:$PATH”

Change to older version of CUDA for deepspeed [12.0 to 11.7]
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

conda create -n deepspeed python=3.10 anaconda
conda activate deepspeed

conda init bash

export LD_LIBRARY_PATH=~/anaconda3/envs/deepspeed/lib/:$LD_LIBRARY_PATH
export PATH=~/anaconda3/envs/deepspeed/bin:$PATH

conda install -c anaconda git
git clone https://github.com/microsoft/DeepSpeedExamples.git





cd DeepSpeedExamples/applications/DeepSpeed-Chat

Cuda Error
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
export CUDA_HOME=/usr/local/cuda-11.7

pip install -r requirements.txt


fused_adam error with deepspeed
https://www.youtube.com/watch?v=S2mtGxf2ZEs&ab_channel=TroubleChute 
https://www.deepspeed.ai/tutorials/advanced-install/ 

fix gcc error: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version/46380601#46380601 
MAX_GCC_VERSION=9
sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
sudo ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/bin/gcc 
sudo ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/bin/g++


pip install deepspeed
install cuda ops: https://www.deepspeed.ai/tutorials/advanced-install/ 
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 pip install deepspeed --global-option="build_ext" --global-option="-j8" 




cd training/step1_supervised_finetuning/training_scripts

[check video for how to change wandb commands in py and sh lsfiles]

cd [type of gpu]

move main.py to the same folder

bash [run type]


to solve the no version information available error:

sudo apt upgrade bash


bypass hostfile

error
https://github.com/microsoft/DeepSpeed/issues/3208 
change from 8 to 2 in single node run_1.3b.sh


go to ~/DeepSpeedExamples/applications/DeepSpeed-Chat $ python3 train.py --step 1 --deployment-type single_gpu

https://discuss.huggingface.co/t/issues-with-building-extensions-in-deepspeed/6323 
issue about 

correct project name




VENV Method

$ sudo apt install libaio-dev build-essential
$ python -m venv venv_ds
$ source venv_ds/bin/activate

(venv_ds) $ export TORCH_CUDA_ARCH_LIST="9.0"
(venv_ds) $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(venv_ds) $ DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 pip install --no-cache-dir deepspeed
(venv_ds) $ git clone https://github.com/microsoft/DeepSpeedExamples.git
(venv_ds) $ cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat
(venv_ds) $ pip install -r requirements.txt
(venv_d $ cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
pip install codecarbon
pip install wandb
pip3 install click --upgrade
pip install protobuf==3.19.0
pip install prometheus-client
wandb login
paste new main.py with codecarbon and wandb, project name
open tmux session
(venv_ds) $ bash ./training_scripts/single_node/run_1.3b.sh

step2_reward_model_finetuning

see logs
tail -f ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/training.log




Dayuan Code
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm ~/miniconda3/miniconda.sh && \
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create --name ds_test python==3.9
conda activate ds_test
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install deepspeed
git clone https://github.com/microsoft/DeepSpeedExamples.git

cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat
pip install -r requirements.txt
cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/

ACTOR_MODEL_PATH=~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output
CRITIC_MODEL_PATH=~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output

pip install codecarbon
pip install wandb
pip3 install click --upgrade
pip install protobuf==3.19.0
pip install prometheus-client
wandb login
bash ./training_scripts/single_node/run_1.3b.sh
Open detachable session
-runs regardless of your system
To have a session that you can safely detach from and attach to when needed you can use tmux.
Running this will get you to a new "tmux" session:
$ tmux
from there run your script (replace this with the bash command)
$ python your_script.py
..
..
..
..
to avoid killing the running process you can press CTRL+B and then D to detach from it.
Then you can leave it running in the background.
 
At this point, you can terminate the SSH session without affecting the running command in the detached tmux session.
 
If you want you can attach to this terminal session like this.
List existing tmux sessions:
$ tmux ls
0: 1 windows (created Fri Jun  2 01:57:21 2023) (attached)
session 0 is active, to attach to it:
$ tmux attach -t 0
and this will get you the output originating from your script.
Calculate Emission
https://www.Codecarbon.io 
-

https://mlco2.github.io/codecarbon/usage.html 





































Publication
https://www.igscc.org/igsc23-cfp 
https://www.overleaf.com/project/647f4dc2ac0a72fe012e079a
Ex publication: https://www.sciencedirect.com/science/article/abs/pii/S2210537923000288 
Abstract
reviewers looking for: what problem you are addressing, why does it matter, what is your solution, how to prove effectiveness of solution
problem: gpt models are environmentally costly and we lack understanding of the real cost and what contributes to the cost
solution: in this position paper, we want to show the first hand of real measurements using one of the most recent training frameworks, deepspeed, which uses RLHF to …
test effects of epochs, learning rate, etc. and limit specific ones to reduce as much carbon footprint as possible
also determine the minimum accuracy needed to be acceptable, and cut training time
effectiveness of solution: 
related work: 
https://academia.stackexchange.com/questions/68164/how-to-write-a-related-work-section-in-computer-science 
show state of the art of Green AI, what other people have done, distinguish the contribution of this paper from other works
prevent companies from testing different parameters, each train costs a lot of resources
develop an algorithm that will help 
Position paper - focus more on idea

gpt-4 super big, nobody knows the quantity of impact
training resources, learning rate, decay
foundation model training more resource hungry, carbon intensive
lack of framework/methodology to evaluate carbon footprint, quantitative analysis
role of each parameter, impact on carbon footprint


Comparing different batch sizes for each run

proposal document
-focus on H100

Experiments
test different batch sizes on step1, show that a bigger batch size uses up more memory/energy, but takes up much less time and overall lower carbon footprint
(make minimum on graph) (16 epochs)
go through test 1, 2, 3 with optimal parameters and calculate overall emissions (8 epochs)

Camera Ready Paper

https://www.overleaf.com/learn/latex/Using_colours_in_LaTeX 
https://tex.stackexchange.com/questions/312574/colorbox-does-not-linebreak 
https://docs.wandb.ai/?_gl=1*b2ng2q*_ga*MTM0NDMzNzk3MS4xNjk2MTk3MTQz*_ga_JH1SJHJQXJ*MTY5NjE5NzE0Mi4xLjEuMTY5NjE5ODk4OC42MC4wLjA. 

https://wandb.ai/stacey/presets/reports/Custom-Scatter-Plots--VmlldzoyNjY4NTc 

https://github.com/wandb/wandb/blob/a5599c60564f9567f3e4038fed12cf5fd018ce07/wandb/sdk/internal/system/assets/gpu.py#L268 
