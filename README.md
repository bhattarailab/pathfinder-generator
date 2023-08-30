# Pathfinder Generator

Adapted from: https://github.com/drewlinsley/pathfinder

## Instruction on LRA dataset generator.

This program uses python-2.7.   
Install required libraries.   
```pip install -r requirements.txt``` 

To generate pathfinder-128 with no gap between curves:   
```python LRA_pathfinder.py <number of machines> <machine index> <total number of images to generate> <task> <seed> <OPTIONAL-data directory>```

Task can be either `nogap` or float values for interpolation e.g. `0.5`   
Seed can be between `0` and `100000`.   
Total number of images is divided among machines if multiple machines are used.


## Example Scripts
### Generating nogap dataset
```python LRA_pathfinder.py 1 0 10 nogap 42```

### Generating interpolated dataset
```python LRA_pathfinder.py 1 0 10 0.0 42```   
```python LRA_pathfinder.py 1 0 10 0.7 42```   
```python LRA_pathfinder.py 1 0 10 1.4 42```   

## References
Configuration for Pathfinder Datasets on LRA. [here](https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/pathfinder.py)