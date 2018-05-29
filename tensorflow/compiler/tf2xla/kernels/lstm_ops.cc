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

xla::XlaOp SigmoidGrad(xla::XlaBuilder* builder, DataType dtype, const xla::XlaOp& x) {
  auto one = XlaHelpers::One(builder, dtype);
  return builder->Mul(x, builder->Sub(one, x));
}

xla::XlaOp TanhGrad(xla::XlaBuilder* builder, DataType dtype, const xla::XlaOp& x) {
  auto one = XlaHelpers::One(builder, dtype);
  return builder->Sub(one, builder->Mul(x, x));
}

struct LSTMBlockForwardInput {
  xla::XlaOp x;
  xla::XlaOp cs_prev;
  xla::XlaOp h_prev;
  xla::XlaOp w;
  xla::XlaOp wci;
  xla::XlaOp wcf;
  xla::XlaOp wco;
  xla::XlaOp b;
};

struct LSTMBlockForwardOutput {
  xla::XlaOp i;
  xla::XlaOp cs;
  xla::XlaOp f;
  xla::XlaOp o;
  xla::XlaOp ci;
  xla::XlaOp co;
  xla::XlaOp h;
};

LSTMBlockForwardOutput LSTMBlockCellForward(
    xla::XlaBuilder* builder, DataType dtype, const LSTMBlockForwardInput& input,
    int64 cell_size, float forget_bias, float cell_clip, bool use_peephole) {
  // Concat xh = [x, h]
  xla::XlaOp xh = builder->ConcatInDim({input.x, input.h_prev}, 1);

  // states1 = xh * w + b
  xla::XlaOp icfo = builder->Add(builder->Dot(xh, input.w), input.b, {1});

  // input gate
  xla::XlaOp i = Sigmoid(builder, dtype, builder->SliceInDim(icfo, 0, cell_size, 1, 1));

  // cell input
  xla::XlaOp ci = builder->Tanh(builder->SliceInDim(icfo, cell_size, 2 * cell_size, 1, 1));

  // forget gate
  xla::XlaOp f = Sigmoid(
      builder, dtype,
      builder->Add(
        builder->SliceInDim(icfo, 2 * cell_size, 3 * cell_size, 1, 1),
        XlaHelpers::FloatLiteral(builder, dtype, forget_bias)));

  // cs = ci .* i + f .* cs_prev
  xla::XlaOp cs = builder->Add(
      builder->Mul(ci, i),
      builder->Mul(f, input.cs_prev));

  if (cell_clip > 0.0) {
    cs = builder->Clamp(
      cs,
      XlaHelpers::FloatLiteral(builder, dtype, -cell_clip),
      XlaHelpers::FloatLiteral(builder, dtype, cell_clip));
  }

  xla::XlaOp co = builder->Tanh(cs);

  // output gate
  xla::XlaOp o = Sigmoid(builder, dtype, builder->SliceInDim(icfo, 3 * cell_size, 4 * cell_size, 1, 1));

  xla::XlaOp h = builder->Mul(o, co);

  return LSTMBlockForwardOutput{ i, cs, f, o, ci, co, h };
}

struct LSTMBlockBackwardInput {
  xla::XlaOp x;
  xla::XlaOp cs_prev;
  xla::XlaOp h_prev;
  xla::XlaOp w;
  xla::XlaOp wci;
  xla::XlaOp wcf;
  xla::XlaOp wco;
  xla::XlaOp b;
  xla::XlaOp i;
  xla::XlaOp cs;
  xla::XlaOp f;
  xla::XlaOp o;
  xla::XlaOp ci;
  xla::XlaOp co;
  xla::XlaOp h_grad;
  xla::XlaOp cs_grad;
};

struct LSTMBlockBackwardOutput {
  xla::XlaOp cs_prev_grad;
  xla::XlaOp dicfo;
  xla::XlaOp wci_grad;
  xla::XlaOp wcf_grad;
  xla::XlaOp wco_grad;
};


LSTMBlockBackwardOutput LSTMBlockCellBackward(
    xla::XlaBuilder* builder, DataType dtype,
    const LSTMBlockBackwardInput& input, int64 cell_size, bool use_peephole) {

  xla::XlaOp one = XlaHelpers::One(builder, dtype);

  // do = sigm'(o) * dh * co
  xla::XlaOp do_ = builder->Mul(
      SigmoidGrad(builder, dtype, input.o),
      builder->Mul(input.h_grad, input.co));

  // dcs = tanh'(cs) * dh * o + (dcs[t + 1])
  xla::XlaOp dcs = builder->Add(
      builder->Mul(
        TanhGrad(builder, dtype, input.o),
        builder->Mul(input.h_grad, input.o)),
      input.cs_grad);

  // dci = tanh'(ci) * dcs * i
  xla::XlaOp dci = builder->Mul(
      TanhGrad(builder, dtype, input.ci),
      builder->Mul(dcs, input.i));

  // df[t] = sigm'(f[t]) * dcs[t] * cs[t - 1]
  xla::XlaOp df = builder->Mul(
      SigmoidGrad(builder, dtype, input.f),
      builder->Mul(dcs, input.cs_prev));

  // di[t] = sigm'(i[t]) dcs[t] ci[t]     
  xla::XlaOp di = builder->Mul(
      SigmoidGrad(builder, dtype, input.i),
      builder->Mul(dcs, input.ci));

  xla::XlaOp dicfo = builder->ConcatInDim({di, dci, df, do_}, 1);
  xla::XlaOp cs_prev_grad = builder->Mul(dcs, input.f);

  xla::XlaOp zero = XlaHelpers::Zero(builder, dtype);
  xla::XlaOp wci_grad = builder->Broadcast(zero, {cell_size});
  xla::XlaOp wcf_grad = builder->Broadcast(zero, {cell_size});
  xla::XlaOp wco_grad = builder->Broadcast(zero, {cell_size});

  return LSTMBlockBackwardOutput{ cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad };
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
      const TensorShape wco_shape = ctx->InputShape(6);
      const TensorShape b_shape = ctx->InputShape(7);

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
      xla::XlaOp wco = ctx->Input(6);
      xla::XlaOp b = ctx->Input(7);

      // build the operation now
      auto dtype = input_type(0);

      auto output = LSTMBlockCellForward(
          ctx->builder(), dtype,
          {x, cs_prev, h_prev, w, wci, wcf, wco, b},
          cell_size, forget_bias_, cell_clip_, use_peephole_);

      ctx->SetOutput(0, output.i);
      ctx->SetOutput(1, output.cs);
      ctx->SetOutput(2, output.f);
      ctx->SetOutput(3, output.o);
      ctx->SetOutput(4, output.ci);
      ctx->SetOutput(5, output.co);
      ctx->SetOutput(6, output.h);
    }
  private:
    float forget_bias_;
    float cell_clip_;
    bool use_peephole_;
};

REGISTER_XLA_OP(Name("LSTMBlockCell"), LSTMBlockCellOp);

class LSTMBlockCellGradOp : public XlaOpKernel {
  public:
    explicit LSTMBlockCellGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    }

    void Compile(XlaOpKernelContext* ctx) override {

      // This macro gets the given input index, and defines
      // an alias for the operation, and an alias for the shape.
      #define INPUT_TENSOR(NAME, INDEX) \
        const TensorShape NAME##_shape = ctx->InputShape(INDEX); \
        xla::XlaOp NAME = ctx->Input(INDEX)
    
      INPUT_TENSOR(x, 0);
      INPUT_TENSOR(cs_prev, 1);
      INPUT_TENSOR(h_prev, 2);
      INPUT_TENSOR(w, 3);
      INPUT_TENSOR(wci, 4);
      INPUT_TENSOR(wcf, 5);
      INPUT_TENSOR(wco, 6);
      INPUT_TENSOR(b, 7);
      INPUT_TENSOR(i, 8);
      INPUT_TENSOR(cs, 9);
      INPUT_TENSOR(f, 10);
      INPUT_TENSOR(o, 11);
      INPUT_TENSOR(ci, 12);
      INPUT_TENSOR(co, 13);
      INPUT_TENSOR(cs_grad, 14);
      INPUT_TENSOR(h_grad, 15);

      #undef INPUT_TENSOR

      const int64 batch_size = x_shape.dim_size(0);
      const int64 input_size = x_shape.dim_size(1);
      const int64 cell_size = cs_prev_shape.dim_size(1);

      OP_REQUIRES(ctx, !use_peephole_,
                  errors::InvalidArgument("Xla LSTMBlockCellGrad only supports use_peephole=False."));
      
      // Sanity checks for our input shapes.
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

      // This macro checks that the given input is a matrix batch x cell_size
      #define CHECK_BATCH_SHAPE(INPUT_NAME) \
        OP_REQUIRES(ctx, INPUT_NAME##_shape.dim_size(0) == batch_size, \
                    errors::InvalidArgument( \
                      #INPUT_NAME ".dim_size(0) != batch_size: ", INPUT_NAME##_shape.dim_size(0), \
                      " vs. ", batch_size)); \
        OP_REQUIRES(ctx, INPUT_NAME##_shape.dim_size(1) == cell_size, \
                    errors::InvalidArgument( \
                      #INPUT_NAME ".dim_size(1) != cell_size: ", INPUT_NAME##_shape.dim_size(1), \
                      " vs. ", cell_size))
    
      CHECK_BATCH_SHAPE(cs_prev);
      CHECK_BATCH_SHAPE(h_prev);
      CHECK_BATCH_SHAPE(i);
      CHECK_BATCH_SHAPE(cs);
      CHECK_BATCH_SHAPE(f);
      CHECK_BATCH_SHAPE(o);
      CHECK_BATCH_SHAPE(ci);
      CHECK_BATCH_SHAPE(co);
      CHECK_BATCH_SHAPE(cs_grad);
      CHECK_BATCH_SHAPE(h_grad);

      #undef CHECK_BATCH_SHAPE

      // Start computation
      auto dtype = input_type(0);

      auto output = LSTMBlockCellBackward(
          ctx->builder(), dtype,
          {x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad},
          cell_size, use_peephole_);

      ctx->SetOutput(0, output.cs_prev_grad);
      ctx->SetOutput(1, output.dicfo);
      ctx->SetOutput(2, output.wci_grad);
      ctx->SetOutput(3, output.wcf_grad);
      ctx->SetOutput(4, output.wco_grad);
    }
  private:
    bool use_peephole_;
};

REGISTER_XLA_OP(Name("LSTMBlockCellGrad"), LSTMBlockCellGradOp);

}  // anonymous namespace
}  // namespace tensorflow
