// Accuracy test for xlns16 functions added in PRs #22, #35, #36
// Tests softmax, layernorm, and activation functions against FP32 reference
// Reports max and mean relative error (%) for each function

#define xlns16_ideal
#include "../xlns16.cpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <numeric>

// ---- FP32 reference implementations ----

static void fp32_softmax(const float *a, float *out, int n) {
  float max_val = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] > max_val)
      max_val = a[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    out[i] = expf(a[i] - max_val);
    sum += out[i];
  }
  for (int i = 0; i < n; i++)
    out[i] /= sum;
}

static void fp32_layernorm(const float *x, float *out, const float *gamma,
                           const float *beta, int n, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < n; i++)
    mean += x[i];
  mean /= n;
  float var = 0.0f;
  for (int i = 0; i < n; i++)
    var += (x[i] - mean) * (x[i] - mean);
  var /= n;
  float inv_std = 1.0f / sqrtf(var + eps);
  for (int i = 0; i < n; i++) {
    out[i] = (x[i] - mean) * inv_std;
    if (gamma)
      out[i] *= gamma[i];
    if (beta)
      out[i] += beta[i];
  }
}

// ---- Error helpers ----

static float rel_error(float a, float b) {
  if (fabsf(b) < 1e-9f)
    return fabsf(a) < 1e-9f ? 0.0f : 1.0f;
  return fabsf((a - b) / b);
}

static void report_array_error(const char *name, const float *ref,
                               const float *got, int n) {
  float max_err = 0.0f, sum_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float e = rel_error(got[i], ref[i]);
    if (e > max_err)
      max_err = e;
    sum_err += e;
  }
  printf("%-20s  max_rel_err: %6.3f%%  mean_rel_err: %6.3f%%\n", name,
         max_err * 100.0f, sum_err / n * 100.0f);
}

// ---- Tests ----

void test_softmax_accuracy() {
  printf("\n=== Softmax Accuracy vs FP32 ===\n");

  // Test 1: typical attention logits (mixed values)
  {
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float ref[8], got[8];
    int n = 8;
    fp32_softmax(in, ref, n);

    xlns16 lns_in[8], lns_out[8];
    for (int i = 0; i < n; i++)
      lns_in[i] = fp2xlns16(in[i]);
    xlns16_softmax(lns_in, lns_out, n);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(lns_out[i]);
    report_array_error("softmax (ascending)", ref, got, n);
  }

  // Test 2: peaked distribution (large spread)
  {
    float in[] = {-5.0f, -3.0f, 0.0f, 3.0f, 5.0f, 8.0f, -1.0f, 2.0f};
    float ref[8], got[8];
    int n = 8;
    fp32_softmax(in, ref, n);

    xlns16 lns_in[8], lns_out[8];
    for (int i = 0; i < n; i++)
      lns_in[i] = fp2xlns16(in[i]);
    xlns16_softmax(lns_in, lns_out, n);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(lns_out[i]);
    report_array_error("softmax (spread)", ref, got, n);
  }

  // Test 3: near-uniform (expected: all ~0.125)
  {
    float in[] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float ref[8], got[8];
    int n = 8;
    fp32_softmax(in, ref, n);

    xlns16 lns_in[8], lns_out[8];
    for (int i = 0; i < n; i++)
      lns_in[i] = fp2xlns16(in[i]);
    xlns16_softmax(lns_in, lns_out, n);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(lns_out[i]);
    report_array_error("softmax (uniform)", ref, got, n);
  }
}

void test_layernorm_accuracy() {
  printf("\n=== Layer Norm Accuracy vs FP32 ===\n");

  // Test 1: typical embedding vector
  {
    float in[] = {0.5f, -1.2f, 3.1f, 0.8f, -0.3f, 2.4f, -2.1f, 1.7f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ref[8], got[8];
    int n = 8;
    float eps = 1e-5f;

    fp32_layernorm(in, ref, gamma, beta, n, eps);

    xlns16 lns_in[8], lns_out[8], lns_gamma[8], lns_beta[8];
    for (int i = 0; i < n; i++) {
      lns_in[i] = fp2xlns16(in[i]);
      lns_gamma[i] = fp2xlns16(gamma[i]);
      lns_beta[i] = fp2xlns16(beta[i]);
    }
    xlns16_layernorm(lns_in, lns_out, lns_gamma, lns_beta, n, eps);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(lns_out[i]);
    report_array_error("layernorm (identity)", ref, got, n);
  }

  // Test 2: with learned gamma/beta
  {
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float gamma[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float beta[] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float ref[8], got[8];
    int n = 8;
    float eps = 1e-5f;

    fp32_layernorm(in, ref, gamma, beta, n, eps);

    xlns16 lns_in[8], lns_out[8], lns_gamma[8], lns_beta[8];
    for (int i = 0; i < n; i++) {
      lns_in[i] = fp2xlns16(in[i]);
      lns_gamma[i] = fp2xlns16(gamma[i]);
      lns_beta[i] = fp2xlns16(beta[i]);
    }
    xlns16_layernorm(lns_in, lns_out, lns_gamma, lns_beta, n, eps);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(lns_out[i]);
    report_array_error("layernorm (gamma=0.5,beta=0.1)", ref, got, n);
  }
}

void test_activation_accuracy() {
  printf("\n=== Activation Accuracy vs FP32 ===\n");

  float test_vals[] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f,
                       0.5f,  1.0f,  2.0f,  3.0f,  4.0f};
  int n = 10;

  // Sigmoid: 1 / (1 + exp(-x))
  {
    float ref[10], got[10];
    for (int i = 0; i < n; i++)
      ref[i] = 1.0f / (1.0f + expf(-test_vals[i]));
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(xlns16_sigmoid(fp2xlns16(test_vals[i])));
    report_array_error("sigmoid", ref, got, n);
  }

  // SiLU: x * sigmoid(x)
  {
    float ref[10], got[10];
    for (int i = 0; i < n; i++) {
      float s = 1.0f / (1.0f + expf(-test_vals[i]));
      ref[i] = test_vals[i] * s;
    }
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(xlns16_silu(fp2xlns16(test_vals[i])));
    report_array_error("silu", ref, got, n);
  }

  // Tanh
  {
    float ref[10], got[10];
    for (int i = 0; i < n; i++)
      ref[i] = tanhf(test_vals[i]);
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(xlns16_tanh(fp2xlns16(test_vals[i])));
    report_array_error("tanh", ref, got, n);
  }

  // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
  {
    float ref[10], got[10];
    for (int i = 0; i < n; i++)
      ref[i] = test_vals[i] * 0.5f * (1.0f + erff(test_vals[i] / sqrtf(2.0f)));
    for (int i = 0; i < n; i++)
      got[i] = xlns162fp(xlns16_gelu(fp2xlns16(test_vals[i])));
    report_array_error("gelu", ref, got, n);
  }
}

void test_dot_product_accuracy() {
  printf("\n=== Dot Product Accuracy vs FP32 ===\n");

  // Small vector
  {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    int n = 8;
    // FP32: 1*8+2*7+3*6+4*5+5*4+6*3+7*2+8*1 = 120
    float fp_ref = 0.0f;
    for (int i = 0; i < n; i++)
      fp_ref += a[i] * b[i];
    float lns_result = xlns16_vec_dot_f32(a, b, n);
    printf("dot product (n=8)    fp32: %.4f  xlns16: %.4f  rel_err: %.3f%%\n",
           fp_ref, lns_result, rel_error(lns_result, fp_ref) * 100.0f);
  }

  // Larger vector (simulates a transformer weight row dotted with activations)
  {
    int n = 64;
    float a[64], b[64];
    float fp_ref = 0.0f;
    for (int i = 0; i < n; i++) {
      a[i] = 0.1f * ((i % 7) - 3); // small mixed values like model activations
      b[i] = 0.01f * (i % 5 + 1);  // small positive like Q4 dequantized weights
      fp_ref += a[i] * b[i];
    }
    float lns_result = xlns16_vec_dot_f32(a, b, n);
    printf("dot product (n=64)   fp32: %.4f  xlns16: %.4f  rel_err: %.3f%%\n",
           fp_ref, lns_result, rel_error(lns_result, fp_ref) * 100.0f);
  }
}

int main() {
  printf("=====================================================\n");
  printf("  xlns16 Accuracy Tests: LNS vs FP32 Reference\n");
  printf("  All errors reported as relative %% vs FP32\n");
  printf("=====================================================\n");

  test_softmax_accuracy();
  test_layernorm_accuracy();
  test_activation_accuracy();
  test_dot_product_accuracy();

  printf("\nAll accuracy tests completed.\n");
  return 0;
}
