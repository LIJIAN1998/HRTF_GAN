# HRTF_GAN

First, run:
```python
main.py preprocess --hpc False --tag ari-upscale-4
```
to split the train and validation sample ids

Then to train the model:
```python
main.py train --hpc False --tag ari-upscale-4
```
training parameters can be modified in config.py file

To evaluate on test set by run:
```python
main.py test --hpc False --tag ari-upscale-4
```

To compare with barycentric interpolation run:
```python
main.py barycentric_baseline --hpc False --tag ari-upscale-4
```

To get non-individual HRTF selection run:
```python
main.py hrtf_selection_baseline --hpc False --tag ari-upscale-4
```