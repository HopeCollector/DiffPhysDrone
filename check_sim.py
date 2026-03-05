#!/usr/bin/env python3
"""
仿真环境预飞检测脚本（已迁移至 pytest）

推荐用法:
    pytest tests/test_preflight.py -v
    pytest tests/test_preflight.py -v --sim-timeout 3

向后兼容:
    python3 check_sim.py [--sim-timeout 5]
"""
import os
import sys

os.execvp(
    sys.executable,
    [sys.executable, "-m", "pytest",
     "tests/test_preflight.py", "-v", "--tb=short",
     *sys.argv[1:]],
)

