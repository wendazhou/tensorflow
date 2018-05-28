/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

xla::XlaOp Sigmoid(xla::XlaBuilder* builder, DataType dtype, const xla::XlaOp& x) {
  auto half = XlaHelpers::FloatLiteral(builder, dtype, 0.5);
  return builder->Add(half, builder->Mul(half, builder->Tanh(builder->Mul(half, x))));
}

class LSTMBlockCellOp : public XlaOpKernel {
  public:
    explicit LSTMBlockCellOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    }

    void Compile(XlaOpKernelContext* ctx) override {
      const TensorShape x_shape = ctx->InputShape(0);
      const TensorShape cs_prev_shape = ctx->InputShape(1);
      const TensorShape h_prev_shape = ctx->InputShape(2);
      const TensorShape w_shape = ctx->InputShape(3);
      const TensorShape wci_shape = ctx->InputShape(4);
      const TensorShape wcf_shape = ctx->InputShape(5);
      const TensorShape b_shape = ctx->InputShape(6);

      const int64 batch_size = x_shape.dim_size(0);
      const int64 input_size = x_shape.dim_size(1);
      const int64 cell_size = cs_prev_shape.dim_size(1);

      // for the moment, must have use_peephole = False
      OP_REQUIRES(ctx, !use_peephole_,
                  errors::InvalidArgument("Xla LSTMBlockCell only supports use_peephole=False."));

      // Sanity checks for our input shapes.
      OP_REQUIRES(ctx, cs_prev_shape.dim_size(0) == batch_size,
                  errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                          cs_prev_shape.dim_size(0), " vs. ",
                                          batch_size));

      OP_REQUIRES(ctx, h_prev_shape.dim_size(0) == batch_size,
                  errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                          h_prev_shape.dim_size(0), " vs. ",
                                          batch_size));
      OP_REQUIRES(ctx, h_prev_shape.dim_size(1) == cell_size,
                  errors::InvalidArgument(
                      "h_prev.dims(1) != cell_size: ", h_prev_shape.dim_size(1),
                      " vs. ", cell_size));

      OP_REQUIRES(ctx, w_shape.dim_size(0) == input_size + cell_size,
                  errors::InvalidArgument(
                      "w.dim_size(0) != input_size + cell_size: ",
                      w_shape.dim_size(0), " vs. ", input_size + cell_size));
      OP_REQUIRES(ctx, w_shape.dim_size(1) == cell_size * 4,
                  errors::InvalidArgument(
                      "w.dim_size(1) != cell_size * 4: ", w_shape.dim_size(1),
                      " vs. ", cell_size * 4));

      OP_REQUIRES(ctx, b_shape.dim_size(0) == cell_size * 4,
                  errors::InvalidArgument(
                      "b.dim_size(0) != cell_size * 4: ", b_shape.dim_size(0),
                      " vs. ", cell_size * 4));

      xla::XlaOp x = ctx->Input(0);
      xla::XlaOp cs_prev = ctx->Input(1);
      xla::XlaOp h_prev = ctx->Input(2);
      xla::XlaOp w = ctx->Input(3);
      xla::XlaOp wci = ctx->Input(4);
      xla::XlaOp wcf = ctx->Input(5);
      xla::XlaOp b = ctx->Input(6);

      // build the operation now
      auto dtype = input_type(0);

      // Concat xh = [x, h]
      auto xh = ctx->builder()->ConcatInDim({x, h_prev}, 1);
      
      // states1 = xh * w + b
      auto icfo = ctx->builder()->Add(
          ctx->builder()->Dot(xh, w), b, {1});
      
      // input gate
      auto i = Sigmoid(
          ctx->builder(), dtype,
          ctx->builder()->Slice(icfo, {0, 0}, {batch_size, cell_size}, {1, 1}));
        
      // cell input
      auto ci = ctx->builder()->Tanh(
          ctx->builder()->Slice(icfo, {0, cell_size}, {batch_size, cell_size}, {1, 1}));
    
      // forget gate
      auto f = Sigmoid(
          ctx->builder(), dtype,
          ctx->builder()->Add(
            ctx->builder()->Slice(icfo, {0, 2 * cell_size}, {batch_size, cell_size}, {1, 1}),
            XlaHelpers::FloatLiteral(ctx->builder(), dtype, forget_bias_)));

      // cs = ci .* i + f .* cs_prev
      auto cs = ctx->builder()->Add(
          ctx->builder()->Mul(ci, i),
          ctx->builder()->Mul(f, cs_prev));
      
      if (cell_clip_ > 0.0) {
        cs = ctx->builder()->Clamp(
            cs,
            XlaHelpers::FloatLiteral(ctx->builder(), dtype, -cell_clip_),
            XlaHelpers::FloatLiteral(ctx->builder(), dtype, cell_clip_));
      }
      
      auto co = ctx->builder()->Tanh(cs);

      // output gate
      auto o = Sigmoid(
          ctx->builder(), dtype,
          ctx->builder()->Slice(icfo, {0, 3 * cell_size}, {batch_size, cell_size}, {1, 1}));
      
      auto h = ctx->builder()->Mul(o, co);

      ctx->SetOutput(0, i);
      ctx->SetOutput(1, cs);
      ctx->SetOutput(2, f);
      ctx->SetOutput(3, o);
      ctx->SetOutput(4, ci);
      ctx->SetOutput(5, co);
      ctx->SetOutput(6, h);
    }
  private:
    float forget_bias_;
    float cell_clip_;
    bool use_peephole_;
};

REGISTER_XLA_OP(Name("LSTMBlockCell"), LSTMBlockCellOp)

}  // anonymous namespace
}  // namespace tensorflow