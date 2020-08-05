# Hangul Single Text Recognition model

## Structure

This repository uses pytorch-lightning hevaily.

* `model`
* `dataset`
* `config`
* `static`
* `utils`

In `model/`, there are two models now: SimpleCNN and SATNet. You can specify which model you want to use with the arguments.

There are no straightforward inference / visualization options yet.

## Instructions

You first need a dataset to run HaSTeR. Refer to [KoTDG](https://github.com/Diuven/KoTDG) for details.  
Dataset should be in the following format:

* `<name>`
  * `train`
    * `%d_%s.jpg % (index, text)` where index is interger and text is single hangul character. (e.g. `3885_알.jpg`)
  * `valid`
    * same as `train`
  * `tests`
    * same as `train`

---

Install all python packages written in `requirements.txt`.  
You might need to use venv, conda or Docker.

Execute the following to train the NN.

```bash
python3 run.py --config config/default --model
```

Run `python3 run.py -h` for a small help.

The logs and checkpoints will be save in `logs` folder.

### Real World Data

If you want to train or test on real world data, you can use `utils/scale.py` to rescale & rename files.

Refer to `utils/scale.py` about the usage. It is recommended to process data in `static/files/` directory.

Example: `python utils/scale.py static/files/raw/*.jpg --out_path static/files/processed`

---

## Reference

<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning>
