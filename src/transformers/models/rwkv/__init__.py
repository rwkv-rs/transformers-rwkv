# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_rwkv import *
    from .modeling_rwkv import *
    from .rwkv7.configuration_rwkv7 import *
    from .rwkv7.modeling_rwkv7 import *
else:
    import sys

    _file = globals()["__file__"]
    # Include rwkv7 submodule in the lazy import structure
    _import_structure = define_import_structure(_file)
    _rwkv7_structure = define_import_structure(
        __path__[0] + "/rwkv7", prefix="rwkv7"
    )
    for key, value in _rwkv7_structure.items():
        if key in _import_structure:
            _import_structure[key].update(value)
        else:
            _import_structure[key] = value
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
