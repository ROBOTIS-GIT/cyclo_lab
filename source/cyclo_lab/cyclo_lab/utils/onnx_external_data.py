# Copyright 2026 ROBOTIS CO., LTD.
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
#
# Author: Kiwoong Park

"""Keep exported ONNX policies self-contained for deployment."""

import os
import tempfile


def embed_onnx_external_data(model_path: str) -> None:
    """Embed external weights and remove the exporter's standard data sidecar."""
    import onnx

    metadata = onnx.load(model_path, load_external_data=False)
    # Include tensors in nested graphs and attributes as well as initializers.
    def contains_external_data(message):
        if isinstance(message, onnx.TensorProto) and message.external_data:
            return True
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                messages = value if field.is_repeated else (value,)
                if any(contains_external_data(child) for child in messages):
                    return True
        return False

    if not contains_external_data(metadata):
        return

    model = onnx.load(model_path, load_external_data=True)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    fd, temporary_path = tempfile.mkstemp(prefix=".onnx-embedded-", suffix=".onnx", dir=model_dir)
    os.close(fd)
    try:
        onnx.save_model(model, temporary_path, save_as_external_data=False)
        embedded = onnx.load(temporary_path, load_external_data=False)
        if contains_external_data(embedded):
            raise RuntimeError(f"Failed to embed ONNX external data: {model_path}")
        onnx.checker.check_model(embedded, full_check=True)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    # Only clean up the standard sidecar owned by this policy export.
    sidecar_path = f"{model_path}.data"
    if os.path.isfile(sidecar_path):
        os.remove(sidecar_path)
    print(f"[INFO]: Embedded ONNX weights into a single file: {model_path}", flush=True)
