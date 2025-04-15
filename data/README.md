# Expected use of this directory

This directory is designed to hold the preprocessed data (data of the form `hand1.json`) for this project.

# Expected setup for this directory

By default, the `trainer` expects the following structure of this directory:

```
 data
├── 󰂺 README.md
├──  training_set
│   ├──  metadata.csv
│   └──  jsons
│       ├──  (filenames_list[0]).json
│       ├──  (filenames_list[1]).json
│       └── ...
└──  validation_set
    ├──  metadata.csv
    └──  jsons
        └──  (filenames_list[0]).json
        ├──  (filenames_list[1]).json
        └── ...
```

Explained as follows:

- The subdirectories `training_set` and `validation_set` holds
  datas for training and validation respectively.
- The `metadata.csv` is a csv file with 3 columns in order `filename`, `gto`, `human`
  (expects explicit column header, separated by comma),
  specifying one game history training file, and its associated GTO, human gameplay EV values.
- The `jsons` directories holds the files specified in the `metadata.csv` files.

## Example

Suppose the training `metadata.csv` is:

|filename|gto|human|
|-:|:-|:-|
|alice.json|3.147|-1.2|
|bob.json|4.174|4.2069|

or in plain text:

```
filename,gto,human
"alice.json",3.147,-1.2
"bob.json",4.174,4.2069
```

then the `data/training_set/jsons/` directory should have the `alice.json`, `bob.json` files,
and during the training phase, these jsons will be loaded as training data.

# In case you need to change things up

TBD
