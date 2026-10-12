// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/sequence_at_fusion.h"

#include <cassert>
#include <gsl/gsl>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// ---------------------------------------------------------------------------
// CreateOrValidateOnQnn – forward declaration and convenience macros
// ---------------------------------------------------------------------------

#define ValidateOnQnn(qmw, seq_construct_unit, seq_at_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_at_unit), true)
#define CreateOnQnn(qmw, seq_construct_unit, seq_at_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_at_unit), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_at_unit,
                                         bool validate);

// ---------------------------------------------------------------------------
// Helper: check whether an ONNX element data type is representable in QNN
// ---------------------------------------------------------------------------

static bool IsElementTypeSupported(ONNXTensorElementDataType etype) {
  switch (etype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return false;
    default:
      return true;
  }
}

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------

std::unique_ptr<IQnnNodeGroup> SequenceAtFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& seq_construct_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Step 1: confirm we are starting from a SequenceConstruct SingleNode.
  if (seq_construct_unit.OpType() != "SequenceConstruct" ||
      seq_construct_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Step 2: Find the unique SequenceAt consumer of SequenceConstruct's sequence output.
  //
  // GetConsumers() on a sequence-typed ValueInfo is unreliable — the EP plugin API
  // only enumerates consumers of tensor-typed edges. The sequence output of
  // SequenceConstruct is seq(tensor(T)), so GetConsumers() silently returns an empty
  // list. Instead, get the sequence output name from the raw Node API and scan
  // node_to_node_unit for a SequenceAt node whose first (sequence-typed) input matches.
  const Ort::ConstNode sc_node(&seq_construct_unit.GetNode());
  const auto sc_outputs = sc_node.GetOutputs();
  if (sc_outputs.size() != 1) {
    return nullptr;
  }
  const std::string seq_output_name = sc_outputs[0].GetName();

  const OrtNodeUnit* seq_at_unit_ptr = nullptr;
  for (const auto& [node_ptr, unit_ptr] : node_to_node_unit) {
    if (unit_ptr == nullptr || unit_ptr->OpType() != "SequenceAt") continue;
    if (unit_ptr->UnitType() != OrtNodeUnit::Type::SingleNode) continue;
    // Use the raw Node API: seq_at.Inputs()[0] has name="" because ParseOrtValueInfo
    // fails for sequence types. GetInputs() on the underlying OrtNode returns the real names.
    const Ort::ConstNode sa_node(node_ptr);
    const auto sa_raw_inputs = sa_node.GetInputs();
    if (sa_raw_inputs.empty() || sa_raw_inputs[0].GetName() != seq_output_name) continue;
    if (seq_at_unit_ptr != nullptr) {
      return nullptr;  // multiple SequenceAt consumers — cannot fuse
    }
    seq_at_unit_ptr = unit_ptr;
  }
  if (seq_at_unit_ptr == nullptr) {
    return nullptr;
  }

  // Step 3: make sure neither node unit is already claimed by another group.
  if (node_unit_to_qnn_node_group.count(&seq_construct_unit) != 0 ||
      node_unit_to_qnn_node_group.count(seq_at_unit_ptr) != 0) {
    return nullptr;
  }

  // Step 4: validate all N element-tensor inputs of SequenceConstruct.
  const auto& sc_inputs = seq_construct_unit.Inputs();
  const size_t N = sc_inputs.size();
  if (N == 0) {
    return nullptr;
  }

  // All inputs must have an element type supported by QNN.
  for (size_t i = 0; i < N; ++i) {
    if (!IsElementTypeSupported(sc_inputs[i].type)) {
      return nullptr;
    }
  }

  // All inputs must share the same static shape.
  std::vector<uint32_t> elem_shape;
  if (!qnn_model_wrapper.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return nullptr;
  }
  for (size_t i = 1; i < N; ++i) {
    std::vector<uint32_t> shape_i;
    if (!qnn_model_wrapper.GetOnnxShape(sc_inputs[i].shape, shape_i)) {
      return nullptr;
    }
    if (shape_i != elem_shape) {
      return nullptr;
    }
  }

  // Step 5: validate position input of SequenceAt.
  // SequenceAt inputs: [input_sequence (seq), position (scalar int32/int64)].
  // The first input is the sequence-typed edge; the second is the position.
  const auto& sa_inputs = seq_at_unit_ptr->Inputs();
  if (sa_inputs.size() < 2) {
    return nullptr;
  }
  const auto& pos_input = sa_inputs[1];
  if (pos_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      pos_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return nullptr;
  }
  // Position must be scalar (rank-0 tensor).
  // Check the raw ONNX shape directly: a rank-0 scalar has an empty shape vector {}.
  // Do NOT use GetOnnxShape here — it converts {} to {1} for QNN, which would
  // incorrectly flag a valid scalar as a non-scalar 1D tensor.
  // If shape is unknown (nullopt), accept it and let QNN validation catch errors.
  if (pos_input.shape.has_value() && !pos_input.shape->empty()) {
    return nullptr;
  }

  // Step 6: QNN op validation (CreateOrValidateOnQnn in validate=true mode).
  auto group = std::make_unique<SequenceAtFusion>(seq_construct_unit, *seq_at_unit_ptr);
  if (!group->IsSupported(qnn_model_wrapper, logger).IsOK()) {
    return nullptr;
  }
  return group;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SequenceAtFusion::SequenceAtFusion(const OrtNodeUnit& seq_construct_unit,
                                   const OrtNodeUnit& seq_at_unit)
    : node_units_{&seq_construct_unit, &seq_at_unit} {}

// ---------------------------------------------------------------------------
// IQnnNodeGroup interface
// ---------------------------------------------------------------------------

gsl::span<const OrtNodeUnit* const> SequenceAtFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* SequenceAtFusion::GetTargetNodeUnit() const {
  // Use SequenceConstruct as the target so the sort-by-target-node logic in
  // GetQnnNodeGroupsImpl emits this group at the right topological position.
  return node_units_[0];
}

Ort::Status SequenceAtFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Ort::Status SequenceAtFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

// ---------------------------------------------------------------------------
// CreateOrValidateOnQnn
//
// Emits:
//   For each input ti (i = 0..N-1):
//     Reshape(ti) -> [1, d0..dk]       named: _seq_reshape_<base>_<i>
//   Concat([reshape_0..reshape_N-1], axis=0) -> [N, d0..dk]
//                                        named: _seq_stacked_<base>
//   [Cast(int64 -> int32) if needed]
//   [For static negative pos: normalize at build time]
//   [For dynamic pos: Add + Less + Select subgraph]
//   Gather(stacked, pos, axis=0) -> [1, d0..dk]
//                                        named: _seq_gather_out_<base>
//   Reshape([1, d0..dk]) -> [d0..dk]   = the registered output of SequenceAt
//
// Note: QNN Gather requires rank-1 indices ({1} shape). ONNX Gather allows
// rank-0 (scalar) indices, but QNN rejects them with error 3110. The extra
// leading-1 dimension is removed by the trailing Reshape.
// ---------------------------------------------------------------------------

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_at_unit,
                                         bool validate) {
  const auto& sc_inputs = seq_construct_unit.Inputs();
  const size_t N = sc_inputs.size();
  assert(N >= 1);

  // Retrieve element shape from the first input.
  std::vector<uint32_t> elem_shape;
  if (!qmw.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return MAKE_EP_FAIL("SequenceAtFusion: cannot get element shape.");
  }

  // SequenceAt inputs: [sequence (skipped), position].
  const auto& sa_inputs = seq_at_unit.Inputs();
  assert(sa_inputs.size() >= 2);
  const OrtNodeUnitIODef& pos_input = sa_inputs[1];
  const auto& sa_outputs = seq_at_unit.Outputs();
  assert(!sa_outputs.empty());
  const OrtNodeUnitIODef& final_output = sa_outputs[0];

  // Retrieve element data type info from first SC input.
  TensorInfo elem_info{};
  auto elem_info_status = qmw.GetTensorInfo(sc_inputs[0], elem_info);
  if (!elem_info_status.IsOK()) return std::move(elem_info_status);

  // Unique name base derived from SequenceConstruct.
  const std::string base = utils::UniqueNameGenerator().New(seq_construct_unit);

  // ------------------------------------------------------------------
  // Validate mode: probe QNN with a minimal Gather node.
  // ------------------------------------------------------------------
  if (validate) {
    std::vector<uint32_t> stacked_shape;
    stacked_shape.reserve(1 + elem_shape.size());
    stacked_shape.push_back(static_cast<uint32_t>(N));
    stacked_shape.insert(stacked_shape.end(), elem_shape.begin(), elem_shape.end());

    // Stacked input tensor.
    QnnTensorWrapper stacked_tensor(
        base + "_validate_stacked",
        QNN_TENSOR_TYPE_NATIVE,
        elem_info.qnn_data_type,
        elem_info.quant_param.Copy(),
        std::vector<uint32_t>(stacked_shape));

    // Index tensor: scalar int32.
    QnnTensorWrapper index_tensor(
        base + "_validate_index",
        QNN_TENSOR_TYPE_NATIVE,
        QNN_DATATYPE_INT_32,
        QnnQuantParamsWrapper(),
        std::vector<uint32_t>{1u});  // rank-0 (scalar)

    // Gather(data=[N,d..], idx=[1], axis=0) produces output=[1,d..].
    std::vector<uint32_t> gather_out_shape;
    gather_out_shape.push_back(1u);
    gather_out_shape.insert(gather_out_shape.end(), elem_shape.begin(), elem_shape.end());

    bool is_graph_output = qmw.IsGraphOutput(final_output.name);
    Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensor(
        base + "_validate_out",
        out_type,
        elem_info.qnn_data_type,
        elem_info.quant_param.Copy(),
        std::vector<uint32_t>(gather_out_shape));

    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 0;
    QnnParamWrapper axis_param(seq_construct_unit.Index(), seq_construct_unit.Name(),
                               QNN_OP_GATHER_PARAM_AXIS, axis_scalar);

    auto validate_status = qmw.ValidateQnnNode(
        base + "_validate_gather",
        QNN_OP_PACKAGE_NAME_QTI_AISW,
        QNN_OP_GATHER,
        {stacked_tensor.GetQnnTensor(),
         index_tensor.GetQnnTensor()},
        {output_tensor.GetQnnTensor()},
        {axis_param.GetQnnParam()});
    return std::move(validate_status);
  }

  // ------------------------------------------------------------------
  // Full AddToModelBuilder path
  // ------------------------------------------------------------------

  // Step 1: Register each element-tensor input as a QNN tensor.
  for (size_t i = 0; i < N; ++i) {
    if (!qmw.IsQnnTensorWrapperExist(sc_inputs[i].name)) {
      QnnTensorWrapper elem_tensor;
      auto s = qmw.MakeTensorWrapper(sc_inputs[i], elem_tensor);
      if (!s.IsOK()) return std::move(s);
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(elem_tensor)),
                    "SequenceAtFusion: failed to add element tensor.");
    }
  }

  // Step 2: Handle position dtype and normalize.
  TensorInfo pos_info{};
  auto pos_info_status = qmw.GetTensorInfo(pos_input, pos_info);
  if (!pos_info_status.IsOK()) return std::move(pos_info_status);

  std::string pos32_name;

  // Helper: register a rank-1 [1] int32 static constant.
  auto AddInt32Const = [&](const std::string& name, int32_t value) -> Ort::Status {
    if (!qmw.IsQnnTensorWrapperExist(name)) {
      std::vector<uint8_t> data(sizeof(int32_t));
      std::memcpy(data.data(), &value, sizeof(int32_t));
      QnnTensorWrapper tw(name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u}, std::move(data));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add int32 constant.");
    }
    return Ort::Status();
  };

  const bool is_static_pos = pos_info.is_initializer;

  if (is_static_pos) {
    // Static position: normalize at build time (handle negative indices).
    std::vector<uint8_t> raw_bytes;
    auto unpack_s = qmw.UnpackInitializerData(pos_info.initializer_tensor, raw_bytes);
    if (!unpack_s.IsOK()) return std::move(unpack_s);

    int32_t pos_val = 0;
    if (pos_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      int64_t val64 = 0;
      if (raw_bytes.size() >= sizeof(int64_t)) std::memcpy(&val64, raw_bytes.data(), sizeof(int64_t));
      pos_val = static_cast<int32_t>(val64);
    } else {
      if (raw_bytes.size() >= sizeof(int32_t)) std::memcpy(&pos_val, raw_bytes.data(), sizeof(int32_t));
    }

    if (pos_val < 0) pos_val += static_cast<int32_t>(N);
    if (!(pos_val >= 0 && static_cast<size_t>(pos_val) < N)) {
      return MAKE_EP_FAIL("SequenceAtFusion: static position is out of range.");
    }

    pos32_name = base + "_pos_normalized";
    auto add_s = AddInt32Const(pos32_name, pos_val);
    if (!add_s.IsOK()) return std::move(add_s);

  } else {
    // Dynamic position.
    if (pos_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      if (!qmw.IsQnnTensorWrapperExist(pos_input.name)) {
        QnnTensorWrapper pos_tensor;
        auto s = qmw.MakeTensorWrapper(pos_input, pos_tensor);
        if (!s.IsOK()) return std::move(s);
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(pos_tensor)),
                      "SequenceAtFusion: failed to add int64 position tensor.");
      }
      const std::string pos32_cast_name = base + "_pos_int32";
      RETURN_IF_ERROR(qmw.AddCastNode(
          utils::UniqueNameGenerator().New(pos_input.name, QNN_OP_CAST),
          pos_input.name, pos32_cast_name, QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_INT_32, QnnQuantParamsWrapper(), std::vector<uint32_t>{1u},
          /*do_op_validation=*/false));
      pos32_name = pos32_cast_name;
    } else {
      if (!qmw.IsQnnTensorWrapperExist(pos_input.name)) {
        QnnTensorWrapper pos_tensor;
        auto s = qmw.MakeTensorWrapper(pos_input, pos_tensor);
        if (!s.IsOK()) return std::move(s);
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(pos_tensor)),
                      "SequenceAtFusion: failed to add int32 position tensor.");
      }
      pos32_name = pos_input.name;
    }

    // Runtime normalization subgraph for negative indices:
    //   pos_plus_N     = ElementWiseAdd(pos32, N)
    //   is_negative    = ElementWiseLess(pos32, 0)
    //   normalized_pos = ElementWiseSelect(is_negative, pos_plus_N, pos32)
    const std::string n_const_name = base + "_seq_len_N";
    const std::string zero_const_name = base + "_pos_zero";
    RETURN_IF_ERROR(AddInt32Const(n_const_name, static_cast<int32_t>(N)));
    RETURN_IF_ERROR(AddInt32Const(zero_const_name, 0));

    const std::string pos_plus_n_name = base + "_pos_plus_N";
    if (!qmw.IsQnnTensorWrapperExist(pos_plus_n_name)) {
      QnnTensorWrapper tw(pos_plus_n_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add pos_plus_N tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_ADD),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                    {pos32_name, n_const_name}, {pos_plus_n_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseAdd node.");

    const std::string is_neg_name = base + "_is_negative";
    if (!qmw.IsQnnTensorWrapperExist(is_neg_name)) {
      QnnTensorWrapper tw(is_neg_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_BOOL_8,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add is_negative tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_LESS),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_LESS,
                                    {pos32_name, zero_const_name}, {is_neg_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseLess node.");

    const std::string norm_pos_name = base + "_pos_normalized";
    if (!qmw.IsQnnTensorWrapperExist(norm_pos_name)) {
      QnnTensorWrapper tw(norm_pos_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add normalized_pos tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_SELECT),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_SELECT,
                                    {is_neg_name, pos_plus_n_name, pos32_name}, {norm_pos_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseSelect node.");
    pos32_name = norm_pos_name;
  }

  // Step 3: Reshape each input to [1, d0..dk].
  std::vector<uint32_t> reshaped_shape;
  reshaped_shape.reserve(1 + elem_shape.size());
  reshaped_shape.push_back(1u);
  reshaped_shape.insert(reshaped_shape.end(), elem_shape.begin(), elem_shape.end());

  std::vector<std::string> reshape_names;
  reshape_names.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const std::string reshape_out_name = base + "_seq_reshape_" + std::to_string(i);
    if (!qmw.IsQnnTensorWrapperExist(reshape_out_name)) {
      TensorInfo elem_info_i{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(sc_inputs[i], elem_info_i));
      QnnTensorWrapper tw(reshape_out_name, QNN_TENSOR_TYPE_NATIVE,
                          elem_info_i.qnn_data_type, elem_info_i.quant_param.Copy(),
                          std::vector<uint32_t>(reshaped_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add reshape output tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_reshape_" + std::to_string(i), QNN_OP_RESHAPE),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
            {sc_inputs[i].name}, {reshape_out_name}, {}),
        ("SequenceAtFusion: failed to create Reshape node for input " + std::to_string(i)).c_str());
    reshape_names.push_back(reshape_out_name);
  }

  // Step 4: Concat over all reshaped tensors -> [N, d0..dk].
  std::vector<uint32_t> stacked_shape;
  stacked_shape.reserve(1 + elem_shape.size());
  stacked_shape.push_back(static_cast<uint32_t>(N));
  stacked_shape.insert(stacked_shape.end(), elem_shape.begin(), elem_shape.end());
  const std::string stacked_name = base + "_seq_stacked";

  if (!qmw.IsQnnTensorWrapperExist(stacked_name)) {
    TensorInfo elem_info_0{};
    RETURN_IF_ERROR(qmw.GetTensorInfo(sc_inputs[0], elem_info_0));
    QnnTensorWrapper tw(stacked_name, QNN_TENSOR_TYPE_NATIVE,
                        elem_info_0.qnn_data_type, elem_info_0.quant_param.Copy(),
                        std::vector<uint32_t>(stacked_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                  "SequenceAtFusion: failed to add stacked tensor.");
  }
  std::vector<std::string> concat_param_names;
  {
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_scalar.uint32Value = 0u;
    QnnParamWrapper axis_param(seq_construct_unit.Index(),
                               seq_construct_unit.Name() + "_concat",
                               QNN_OP_CONCAT_PARAM_AXIS, axis_scalar);
    const std::string param_name = axis_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axis_param)),
                  "SequenceAtFusion: failed to add Concat axis param.");
    concat_param_names.push_back(param_name);
  }
  RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_CONCAT),
                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                  std::vector<std::string>(reshape_names),
                                  {stacked_name}, std::move(concat_param_names)),
                "SequenceAtFusion: failed to create Concat node.");

  // Step 5: Gather(stacked, pos32, axis=0) -> [1, d0..dk], then Reshape -> [d0..dk].
  //
  // QNN Gather requires rank-1 indices, so the output has an extra leading-1 dim
  // compared to what ONNX Gather with scalar index would produce. The trailing
  // Reshape squeezes it back to elem_shape.
  std::vector<uint32_t> gather_out_shape;
  gather_out_shape.push_back(1u);
  gather_out_shape.insert(gather_out_shape.end(), elem_shape.begin(), elem_shape.end());
  const std::string gather_out_name = base + "_gather_out";

  if (!qmw.IsQnnTensorWrapperExist(gather_out_name)) {
    TensorInfo gather_out_info{};
    auto s = qmw.GetTensorInfo(sc_inputs[0], gather_out_info);
    if (!s.IsOK()) return std::move(s);
    QnnTensorWrapper gather_tw(gather_out_name, QNN_TENSOR_TYPE_NATIVE,
                               gather_out_info.qnn_data_type, gather_out_info.quant_param.Copy(),
                               std::vector<uint32_t>(gather_out_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(gather_tw)),
                  "SequenceAtFusion: failed to add gather_out tensor.");
  }
  std::vector<std::string> gather_param_names;
  {
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 0;
    QnnParamWrapper axis_param(seq_at_unit.Index(), seq_at_unit.Name(),
                               QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
    const std::string param_name = axis_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axis_param)),
                  "SequenceAtFusion: failed to add Gather axis param.");
    gather_param_names.push_back(param_name);
  }
  RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(seq_at_unit),
                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER,
                                  {stacked_name, pos32_name},
                                  {gather_out_name}, std::move(gather_param_names)),
                "SequenceAtFusion: failed to create Gather node.");

  // Reshape [1, d0..dk] -> [d0..dk] to produce the registered SequenceAt output.
  // Use elem_info/elem_shape
  // from the SequenceConstruct inputs instead (validated homogeneous above).
  const bool is_graph_output = qmw.IsGraphOutput(final_output.name);
  Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    QnnTensorWrapper out_tw(final_output.name, out_type,
                            elem_info.qnn_data_type, elem_info.quant_param.Copy(),
                            std::vector<uint32_t>(elem_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                  "SequenceAtFusion: failed to add final output tensor.");
  }
  RETURN_IF_NOT(
      qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base + "_reshape_out", QNN_OP_RESHAPE),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                        {gather_out_name}, {final_output.name}, {}),
      "SequenceAtFusion: failed to create final Reshape node.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
