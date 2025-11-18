
Python version: `3.11`

```Console
$ python3 -m venv GNNca
$ source GNNca/bin/activate
```

To install all the required packages, kindly run
```Console
$ chmod +x install.sh
$ ./install.sh
```

### Run codes

To run a simple test model, just:
```python
python main.py --dataset=cora --gpu=0 --n_runs=10
```

To run all the methods and all codes with logs stored in `./log`:
```Console
$ chmod +x run_all_one.sh
$ ./run_all_one.sh
```
In order to customize your settings, kindly change the parameters within `./config` folder.

The traning logs for WATS on GCN backbond are provided in `./log`.
