# Expected use of this directory

This directory is designed to hold the preprocessed data (data of the form `hand1.json`) for this project.

# Expected setup for this directory

By default, the `trainer` expects the following structure of this directory:

```
 data
├── 󰂺 README.md
├──  gto
│   ├──  *.txt
└──  human
    ├──  metadata.csv
    └──  *.json
```

Explained as follows:

- The subdirectories `gto` and `human` holds datas
  for GTO and human data.
- The `metadata.csv` is a csv file with 3 columns in order `filename`, `ev`
  (expects explicit column header, separated by comma),
  specifying one game history training file
  (in the same directory)
  and its associated human gameplay EV value.

## Example

Suppose the `metadata.csv` in `human` is:

|filename|gto|human|
|-:|:-|:-|
|alice.json|3.147|-1.2|
|bob.json|4.174|4.2069|

or in plain text:

```
filename,ev
"alice.json",3.147
"bob.json",4.174
```

then the `data/human` directory should have the `alice.json`, `bob.json` files.

# In case you need to change things up

TBD
