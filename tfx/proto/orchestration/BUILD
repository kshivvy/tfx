# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tfx:tfx.bzl", "tfx_py_proto_library")

tfx_py_proto_library(
    name = "execution_result_py_pb2",
    srcs = ["execution_result.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "executor_invocation_py_pb2",
    srcs = ["executor_invocation.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "pipeline_py_pb2",
    srcs = ["pipeline.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "local_deployment_config_py_pb2",
    srcs = ["local_deployment_config.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)
