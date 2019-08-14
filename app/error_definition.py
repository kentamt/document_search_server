# 
# @file error_definition.py
# 
# @brief 
# 
# @author Kenta Matsui
# @version 0.9.0
# @date 14-Aug. 2019
# @copyright Copyright (c) 2019
# 
from enum import Enum

# Error Difinition
class Error(Enum):
    NO_MODEL=1
    NO_DOCS=2
    NO_CORPUS=3
    WRONG_INPUT=4
    SOMETHING_WRONG=5

