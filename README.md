# Amnesic Probing
This repository contain all the codebase for the paper:

"Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals"

_Note that a previous version of this paper that appeared on arxiv in 2020 was named: "**When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions**", which we changed to the current title to better reflect our contributions._

## General Notes and Considerations
This work contain many moving parts, which are build one on top of the other, therefore the code
also contain many different parts.
In order to save run-times, we tried to save as much as we can to local files.
Furthermore, we use a queue-based software 
([task spooler](https://vicerveza.homeunix.net/~viric/soft/ts/)) 
in order to parallel the (many) experiments.
Note that only the runners scripts inside of the `runs` directory requires them,
but otherwise one can simply run individual runs without this software.

We direct to the following running scripts for the logical order of how one should run 
these experiments. If you care about a specific run please follow the script to the relevant file.

For any question, query regarding the code, or paper, please reach out at `yanaiela@gmail.com`



## Prerequisites
We use python 3.7, and linux machines for all our experiments

Create a virtual environment:
```sh
conda create -n amnesic_probing python=3.7 anaconda
```

## Walk-through Experiments

### Encode Datasets
This step is required in order to encode the texts into vectors, encoded by BERT.
It also saves the tokenized words and labels for the relevant tasks.

`python runs/encode/run_encode.py`


### Running the _Amnesic Probing_
This process runs the amnesic operation only. Meaning it runs the INLP process on the
relevant data, and save all the projection matrices in a folder.

```python runs/core/run_deprobe.py```

### Evaluate
The following script, runs the basic evaluation, where we compute the LM performance, DKL
for the "best" amnesic projection.
```python runs/evaluate/run_eval.py```

In order to compute the LM scores for all of the projections (Figure 2 in the paper), run:
```python runs/evaluate/run_eval_per_dim.py```

Then, to run fine-grained evaluation (the performance per label), run:
```python runs/evaluate/run_specific_eval.py```

### Per-Layer Runs
Finally, in order to run the final part of the paper (Section 7), there are multiple steps.
Note, that this part takes all the encoding from some layer i, then runs the (pre computed in 
a former step) projection matrix (that does the amnesic operation) on that layer, and then
re-run then encoding from this step forwards.

This is a rather long process (for the training it can take around 8-10 hours on gpu), and
very heavy in disk usage. We save everything to the disk in order to make the further steps faster.
Each encoded vector file (on the train) is about 4G, therefore the encoding of an entire dataset
with all layer-to-layer encoding is about 400G

First, start by encoding:
```python runs/encode/run_layer_encode.py```

Once this step is done (again, it can take a while ~10 hours for each training encoding), run
the evaluation:
```python runs/evaluate/run_layer_wise_lm.py```

```python runs/evaluate/run_layer_wise_deprobe.py```


## Citation
If you find this work relevant to yours, please cite us:
```
@article{amnesic-probing,
    author = {Elazar, Yanai and Ravfogel, Shauli and Jacovi, Alon and Goldberg, Yoav},
    title = "{Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {160-175},
    year = {2021},
    month = {03},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00359},
    url = {https://doi.org/10.1162/tacl\_a\_00359},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00359/1894330/tacl\_a\_00359.pdf},
}

```
