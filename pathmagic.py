# Copyright 2018 Juhan Bae All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path hack to make import statements work."""

import os
import sys

# Explicitly add the file's directory to the path list.
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(os.getcwd(), "modules"))
