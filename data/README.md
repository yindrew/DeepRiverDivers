# Expected use of this directory

This directory is designed to hold the preprocessed data (data of the form `hand1.json`) for this project.

# Expected setup for this directory

By default, the `trainer` expects the following structure of this directory:

```
 data
├── 󰂺 README.md
├──  training_set
│   ├──  filenames_list.pkl
│   ├──  gto_values.pt
│   ├──  human_values.pt
│   └──  jsons
│       ├──  (filenames_list[0]).json
│       ├──  (filenames_list[1]).json
│       └── ...
└──  validation_set
    ├──  filenames_list.pkl
    ├──  gto_values.pt
    ├──  human_values.pt
    └──  jsons
        └──  (filenames_list[0]).json
        ├──  (filenames_list[1]).json
        └── ...
```

Explained as follows:

- The subdirectories `training_set` and `validation_set` holds
  datas for training and validation respectively.
- The `filenames_list.pkl` is a pickle object of list of Python strings,
  containing filenames (including `.json`).
  Expects all filenames in this pickle object to be present in the `jsons` subdirectory.
- `gto_values.pt` and `human_values.pt` specifies the GTO, human EVs,
  corresponding in the same order to `filenames_list.pkl`.

Under the hood, the trainer will find the data based on config parameters,
loads the expected values (either it being `GTO` or `Human`),
loads the json files from `jsons` based on `filenames_list.pkl`,
and use these to construct `torch.utils.data.Dataset` and `DataLoader`.

# In case you need to change things up

TBD
