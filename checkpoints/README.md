# Expected use of this directory

This directory is designed to checkpoints of models.

Checkpoint names using the `ModelConfig.general.["checkpoint_name"]` config parameter.

The directory would look like:

```
 checkpoints
├──  (checkpoint_name)
│   ├──  final.pth
│   └──  best.pth
└── 󰂺 README.md
```

where final and best models are saved to to its corresponding directory.
