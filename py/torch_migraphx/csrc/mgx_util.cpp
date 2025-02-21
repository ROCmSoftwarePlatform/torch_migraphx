/*
 * Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "mgx_util.h"

namespace torch_migraphx {
namespace util {

namespace {
const std::unordered_map<torch::ScalarType, std::string>& torch_mgx_type_map() {
  static const std::unordered_map<torch::ScalarType, std::string> at_mgx_type_map = {
      {torch::kFloat32, "float_type"},
      {torch::kFloat16,  "half_type"},
      {torch::kBFloat16,  "bf16_type"},
      {torch::kFloat64, "double_type"},
      {torch::kInt32, "int32_type"},
      {torch::kInt16, "int16_type"},
      {torch::kInt64, "int64_type"},
      {torch::kInt8, "int8_type"},
      {torch::kUInt8, "uint8_type"},
      {torch::kBool, "bool_type"}};
  return at_mgx_type_map;
}

const std::unordered_map<std::string, torch::ScalarType>& mgx_torch_type_map() {
  static const std::unordered_map<std::string, torch::ScalarType> mgx_at_type_map = {
      {"float_type", torch::kFloat32},
      {"half_type", torch::kFloat16},
      {"bf16_type", torch::kBFloat16},
      {"double_type", torch::kFloat64},
      {"int32_type", torch::kInt32},
      {"int16_type", torch::kInt16},
      {"int64_type", torch::kInt64},
      {"int8_type", torch::kInt8},
      {"uint8_type", torch::kUInt8},
      {"bool_type", torch::kBool}};
  return mgx_at_type_map;
}
} // namespace

const std::unordered_map<torch::ScalarType, std::string>& get_torch_mgx_type_map() {
  return torch_mgx_type_map();
}

const std::unordered_map<std::string, torch::ScalarType>& get_mgx_torch_type_map() {
  return mgx_torch_type_map();
}

torch::ScalarType MGXDataTypeToScalarType(std::string t) {
  auto mgx_torch_type_map = get_mgx_torch_type_map();
  return mgx_torch_type_map.at(t);
}

std::string ScalarTypeToMGXDataType(torch::ScalarType t) {
  auto torch_mgx_type_map = get_torch_mgx_type_map();
  return torch_mgx_type_map.at(t);
}

} // namespace util
} // namespace torch_migraphx