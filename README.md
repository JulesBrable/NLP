# NLP

Repository for the final project of the NLP course (2nd Semester, final year, ENSAE Paris).

## Contents

* The code used to train the machine and deep learning models can be found in the `src` folder (the hyper-parameters used for the finetuning are stored in the `conf` folder), and the `main.py` script orchestrates it all. This `main.py` script can be use to reproduce all empirical results we obtained in my final report (the underlying approach followed in this script is described in my paper).
* `analysis.ipynb` is a notebook that briefly shows how to do the descriptive analysis and use the pretrained models that are described in my paper. This notebook contains 4 main sections : Setup, Descriptive statistics, Experiments and Inference.
* `src` folder contained 7 modules that handle data extraction, visualization, preprocessing, model construction, training, inference and evaluation.

_**Note:** this repository does not contain the final report associated to this study (in which we present more in depth both theoretical and empirical aspects of our work). If you are interested in having a look to this paper, you can ask this paper directly to the author (me)._

## Setup Instructions

From the command line, you will have to follow the following steps to set this project up:

1. Clone this repository:

```bash
git clone https://github.com/JulesBrable/NLP.git
```

2. Go to the project folder:
```bash
cd NLP
```

3. Create and activate conda environnement:
   
```bash
conda create -n nlp python=3.9 -y
conda acitvate nlp
```

4. Install the listed dependencies:
   
```bash
pip install -r requirements.txt
```

## Model Training

To train the model, you can run the following command:

```bash
python main.py --es=True
```

Note that `main.py` can take an additionnal argument : `--es`. This means that by default, we are finetuning the models with an early stopping procedure.

<br>

## Contact

* [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
