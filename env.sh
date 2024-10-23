export TF_CONFIG='{
  "cluster": {
    "worker": ["<IP or hostname of worker0>:12345", "<IP or hostname of worker1>:23456"]
  },
  "task": {"type": "worker", "index": 0}
}'