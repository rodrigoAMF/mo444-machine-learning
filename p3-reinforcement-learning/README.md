# MO444 Project 3 - Reinforcement Learning

O objetivo deste projeto foi desenvolver agente para jogar o jogo PACMAN em 3 layouts diferentes. Para facilitar a execução e a escrita do relatório, o código deste projeto foi dividido em 2 notebooks, `project_3-ga_outputs`, com resultados e comentários sobre a parte 1 do projeto, e project_3-rl_outputs.ipynb, com resultados e comentários sobre a parte 2 do projeto.

## Configuração do ambiente para execução do projeto

Instale o [anaconda](https://www.anaconda.com/products/individual) ou [miniconda](https://docs.conda.io/en/latest/miniconda.html), abra a pasta do projeto em um terminal, e execute os seguintes comandos:

```
conda create -n MO443 python=3.8.5
conda activate MO443
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

Uma vez que esses comandos tenham sido executados, será possível rodar os notebooks do projeto.
