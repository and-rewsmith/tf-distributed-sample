export TF_CONFIG='{
  "cluster": {
    "worker": ["192.168.50.17:12345", "192.168.50.130:23456"]
  },
  "task": {"type": "worker", "index": 0}
}'