You can view a static version (with non-interactive widgets) of the notebook at https://wpreimes.github.io/ml_lecture_final_project/FinalProject.html

# Setup
1) Clone this repository `git clone git@github.com:wpreimes/ml_lecture_final_project.git`
2) CD into the local copy and download the input data (~1 GB) `bash download.sh`
3) Create the environment `conda env create -f environment.yml`
4) Activate it `conda activate ml_lecture`
5) Run the notebook `jupyter lab` and make sure you use the correct interpreter

# Creating a new html version
1) Make sure that jupyter lab saves the widget states: Activate `Settings > Save Widget State Automatically`.
2) Save the notebook and run `jupyter nbconvert --to html FinalProject.ipynb`
