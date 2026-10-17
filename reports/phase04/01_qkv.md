# Observed Phase 4 result: 01_qkv

Actual recorded execution; learner understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase04/01_project_qkv.py",
  "fixture": "hand-set features and projection matrices; no training or linguistic semantics",
  "features": [
    [
      [
        1.0,
        0.0,
        1.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0,
        1.0
      ],
      [
        1.0,
        1.0,
        1.0,
        0.0
      ]
    ]
  ],
  "query": [
    [
      [
        1.0,
        0.0
      ],
      [
        0.0,
        1.0
      ],
      [
        1.0,
        1.0
      ]
    ]
  ],
  "key": [
    [
      [
        1.0,
        0.0
      ],
      [
        0.0,
        1.0
      ],
      [
        1.0,
        0.0
      ]
    ]
  ],
  "value": [
    [
      [
        2.0,
        0.0
      ],
      [
        0.0,
        2.0
      ],
      [
        2.0,
        1.0
      ]
    ]
  ],
  "input_shape": [
    1,
    3,
    4
  ],
  "projection_shapes": [
    [
      1,
      3,
      2
    ],
    [
      1,
      3,
      2
    ],
    [
      1,
      3,
      2
    ]
  ],
  "stored_parameters": 24,
  "gradient_probe": "sum of Q, K and V outputs; not next-token cross-entropy",
  "projection_gradient_norms": {
    "query": 5.099019527435303,
    "key": 5.099019527435303,
    "value": 5.099019527435303
  },
  "final_holdout_accessed": false
}
```
