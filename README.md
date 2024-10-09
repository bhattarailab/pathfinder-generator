# Pathfinder Generator
Adapted from: https://github.com/drewlinsley/pathfinder

A part of Solving LongRangeArena Pathfinder-X: https://github.com/tsumansapkota/Solving-LRA-PathX   

## Instruction on LRA dataset generator.

Install python-2 environment:

* Setup python-2 conda environment.   
`conda env create -f environment.yml`   
`conda activate py2`


ALTERNATIVELY:
* This program uses python-2.7.   
Install required libraries.   
```pip install -r requirements.txt``` 

To generate pathfinder-128 with no gap between curves:   
```python LRA_pathfinder.py <number of machines> <machine index> <total number of images to generate> <task> <seed> <OPTIONAL-data directory>```

Task can be either `nogap` or float values for interpolation e.g. `0.5`   
Seed can be between `0` and `2^32`.   
Total number of images is divided among machines if multiple machines are used.


## Example Scripts
### Generating nogap dataset
```python LRA_pathfinder.py 1 0 10 nogap 42```

### Generating interpolated dataset
```python LRA_pathfinder.py 1 0 10 0.0 42```   
```python LRA_pathfinder.py 1 0 10 0.75 42```   
```python LRA_pathfinder.py 1 0 10 1.5 42```   

## References
Configuration for Pathfinder Datasets on LRA. [here](https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/pathfinder.py)

## Citation
```
@article{sapkota2023dimension,
  title={Dimension Mixer: A Generalized Method for Structured Sparsity in Deep Neural Networks},
  author={Sapkota, Suman and Bhattarai, Binod},
  journal={arXiv preprint arXiv:2311.18735},
  year={2023}
}
```

