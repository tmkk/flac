#include "private/cpu.h"

#ifndef FLAC__INTEGER_ONLY_LIBRARY
#ifndef FLAC__NO_ASM
#if defined FLAC__CPU_ARM && FLAC__HAS_NEONINTRIN
#include "private/lpc.h"
#include "FLAC/assert.h"
#include "FLAC/format.h"
#include <arm_neon.h>

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ < 7
#define vcopyq_laneq_f32 vcopyq_lane_f32
#define vcopyq_laneq_f64 vcopyq_lane_f64
#endif
#endif

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_4(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[])
{
	int i = 0;
	int limit = data_len - 4;
	float32x4_t sum0;

	(void) lag;
	FLAC__ASSERT(lag <= 4);
	FLAC__ASSERT(lag <= data_len);

	sum0 = vdupq_n_f32(0);

	if(limit >= 3) {
		float32x4_t d0, d1;
		float32x4_t sum10, sum20, sum30;
		sum10 = vdupq_n_f32(0);
		sum20 = vdupq_n_f32(0);
		sum30 = vdupq_n_f32(0);
		d0 = vld1q_f32(data);
		for(i = 0; i <= (limit-3); i += 4) {
			float32x4_t d0_orig = d0;
			d1 = vld1q_f32(data+i+4);
			sum0 = vfmaq_laneq_f32(sum0, d0, d0_orig, 0);
			d0 = vcopyq_laneq_f32(d0_orig, 0, d1, 0);
			sum10 = vfmaq_laneq_f32(sum10, d0, d0_orig, 1);
			d0 = vreinterpretq_f32_f64(vcopyq_laneq_f64(vreinterpretq_f64_f32(d0_orig), 0, vreinterpretq_f64_f32(d1), 0));
			sum20 = vfmaq_laneq_f32(sum20, d0, d0_orig, 2);
			d0 = vcopyq_laneq_f32(d1, 3, d0_orig, 3);
			sum30 = vfmaq_laneq_f32(sum30, d0, d0_orig, 3);
			d0 = d1;
		}
		sum0 = vaddq_f32(sum0, vextq_f32(sum10, sum10, 1));
		sum0 = vaddq_f32(sum0, vextq_f32(sum20, sum20, 2));
		sum0 = vaddq_f32(sum0, vextq_f32(sum30, sum30, 3));
	}
	for(; i <= limit; i++) {
		float32x4_t d0;
		d0 = vld1q_f32(data+i);
		sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
	}

	{
		float32x4_t d0 = vdupq_n_f32(0);

		for(; i < (int)data_len; i++) {
			float d = data[i];
			d0 = vextq_f32(d0, d0, 3);
			d0 = vsetq_lane_f32(d, d0, 0);
			sum0 = vfmaq_n_f32(sum0, d0, d);
		}
	}

	vst1q_f32(autoc,   sum0);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_8(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[])
{
	int i = 0;
	int limit = data_len - 8;
	float32x4_t sum0, sum1;

	(void) lag;
	FLAC__ASSERT(lag <= 8);
	FLAC__ASSERT(lag <= data_len);

	sum0 = vdupq_n_f32(0);
	sum1 = vdupq_n_f32(0);

	if(limit >= 3) {
		float32x4_t d0, d1, d2;
		float32x4_t sum10, sum11, sum20, sum21, sum30, sum31;
		sum10 = sum11 = vdupq_n_f32(0);
		sum20 = sum21 = vdupq_n_f32(0);
		sum30 = sum31 = vdupq_n_f32(0); 
		d0 = vld1q_f32(data);
		d1 = vld1q_f32(data+4);
		for(i = 0; i <= (limit-3); i += 4) {
			float32x4_t d0_orig = d0;
			d2 = vld1q_f32(data+i+8);
			sum0 = vfmaq_laneq_f32(sum0, d0, d0_orig, 0);
			sum1 = vfmaq_laneq_f32(sum1, d1, d0_orig, 0);
			d0 = vcopyq_laneq_f32(d0_orig, 0, d2, 0);
			sum10 = vfmaq_laneq_f32(sum10, d0, d0_orig, 1);
			sum11 = vfmaq_laneq_f32(sum11, d1, d0_orig, 1);
			d0 = vreinterpretq_f32_f64(vcopyq_laneq_f64(vreinterpretq_f64_f32(d0_orig), 0, vreinterpretq_f64_f32(d2), 0));
			sum20 = vfmaq_laneq_f32(sum20, d0, d0_orig, 2);
			sum21 = vfmaq_laneq_f32(sum21, d1, d0_orig, 2);
			d0 = vcopyq_laneq_f32(d2, 3, d0_orig, 3);
			sum30 = vfmaq_laneq_f32(sum30, d0, d0_orig, 3);
			sum31 = vfmaq_laneq_f32(sum31, d1, d0_orig, 3);
			d0 = d1;
			d1 = d2;
		}
		sum0 = vaddq_f32(sum0, vextq_f32(sum10, sum11, 1));
		sum1 = vaddq_f32(sum1, vextq_f32(sum11, sum10, 1));
		sum0 = vaddq_f32(sum0, vextq_f32(sum20, sum21, 2));
		sum1 = vaddq_f32(sum1, vextq_f32(sum21, sum20, 2));
		sum0 = vaddq_f32(sum0, vextq_f32(sum30, sum31, 3));
		sum1 = vaddq_f32(sum1, vextq_f32(sum31, sum30, 3));
	}
	for(; i <= limit; i++) {
		float32x4_t d0, d1;
		d0 = vld1q_f32(data+i);
		d1 = vld1q_f32(data+i+4);
		sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
		sum1 = vfmaq_laneq_f32(sum1, d1, d0, 0);
	}

	{
		float32x4_t d0 = vdupq_n_f32(0);
		float32x4_t d1 = vdupq_n_f32(0);

		for(; i < (int)data_len; i++) {
			float d = data[i];
			d1 = vextq_f32(d0, d1, 3);
			d0 = vextq_f32(d0, d0, 3);
			d0 = vsetq_lane_f32(d, d0, 0);
			sum1 = vfmaq_n_f32(sum1, d1, d);
			sum0 = vfmaq_n_f32(sum0, d0, d);
		}
	}

	vst1q_f32(autoc,   sum0);
	vst1q_f32(autoc+4, sum1);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_12(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[])
{
	int i = 0;
	int limit = data_len - 12;
	float32x4_t sum0, sum1, sum2;

	(void) lag;
	FLAC__ASSERT(lag <= 12);
	FLAC__ASSERT(lag <= data_len);

	sum0 = vdupq_n_f32(0);
	sum1 = vdupq_n_f32(0);
	sum2 = vdupq_n_f32(0);

	if(limit >= 3) {
		float32x4_t d0, d1, d2, d3;
		float32x4_t sum10, sum11, sum12, sum20, sum21, sum22, sum30, sum31, sum32;
		sum10 = sum11 = sum12 = vdupq_n_f32(0);
		sum20 = sum21 = sum22 = vdupq_n_f32(0);
		sum30 = sum31 = sum32 = vdupq_n_f32(0); 
		d0 = vld1q_f32(data);
		d1 = vld1q_f32(data+4);
		d2 = vld1q_f32(data+8);
		for(i = 0; i <= (limit-3); i += 4) {
			float32x4_t d0_orig = d0;
			d3 = vld1q_f32(data+i+12);
			sum0 = vfmaq_laneq_f32(sum0, d0, d0_orig, 0);
			sum1 = vfmaq_laneq_f32(sum1, d1, d0_orig, 0);
			sum2 = vfmaq_laneq_f32(sum2, d2, d0_orig, 0);
			d0 = vcopyq_laneq_f32(d0_orig, 0, d3, 0);
			sum10 = vfmaq_laneq_f32(sum10, d0, d0_orig, 1);
			sum11 = vfmaq_laneq_f32(sum11, d1, d0_orig, 1);
			sum12 = vfmaq_laneq_f32(sum12, d2, d0_orig, 1);
			d0 = vreinterpretq_f32_f64(vcopyq_laneq_f64(vreinterpretq_f64_f32(d0_orig), 0, vreinterpretq_f64_f32(d3), 0));
			sum20 = vfmaq_laneq_f32(sum20, d0, d0_orig, 2);
			sum21 = vfmaq_laneq_f32(sum21, d1, d0_orig, 2);
			sum22 = vfmaq_laneq_f32(sum22, d2, d0_orig, 2);
			d0 = vcopyq_laneq_f32(d3, 3, d0_orig, 3);
			sum30 = vfmaq_laneq_f32(sum30, d0, d0_orig, 3);
			sum31 = vfmaq_laneq_f32(sum31, d1, d0_orig, 3);
			sum32 = vfmaq_laneq_f32(sum32, d2, d0_orig, 3);
			d0 = d1;
			d1 = d2;
			d2 = d3;
		}
		sum0 = vaddq_f32(sum0, vextq_f32(sum10, sum11, 1));
		sum1 = vaddq_f32(sum1, vextq_f32(sum11, sum12, 1));
		sum2 = vaddq_f32(sum2, vextq_f32(sum12, sum10, 1));
		sum0 = vaddq_f32(sum0, vextq_f32(sum20, sum21, 2));
		sum1 = vaddq_f32(sum1, vextq_f32(sum21, sum22, 2));
		sum2 = vaddq_f32(sum2, vextq_f32(sum22, sum20, 2));
		sum0 = vaddq_f32(sum0, vextq_f32(sum30, sum31, 3));
		sum1 = vaddq_f32(sum1, vextq_f32(sum31, sum32, 3));
		sum2 = vaddq_f32(sum2, vextq_f32(sum32, sum30, 3));
	}
	for(; i <= limit; i++) {
		float32x4_t d0, d1, d2;
		d0 = vld1q_f32(data+i);
		d1 = vld1q_f32(data+i+4);
		d2 = vld1q_f32(data+i+8);
		sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
		sum1 = vfmaq_laneq_f32(sum1, d1, d0, 0);
		sum2 = vfmaq_laneq_f32(sum2, d2, d0, 0);
	}

	{
		float32x4_t d0 = vdupq_n_f32(0);
		float32x4_t d1 = vdupq_n_f32(0);
		float32x4_t d2 = vdupq_n_f32(0);

		for(; i < (int)data_len; i++) {
			float d = data[i];
			d2 = vextq_f32(d1, d2, 3);
			d1 = vextq_f32(d0, d1, 3);
			d0 = vextq_f32(d0, d0, 3);
			d0 = vsetq_lane_f32(d, d0, 0);
			sum2 = vfmaq_n_f32(sum2, d2, d);
			sum1 = vfmaq_n_f32(sum1, d1, d);
			sum0 = vfmaq_n_f32(sum0, d0, d);
		}
	}

	vst1q_f32(autoc,   sum0);
	vst1q_f32(autoc+4, sum1);
	vst1q_f32(autoc+8, sum2);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_16(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[])
{
	int i = 0;
	int limit = data_len - 16;
	float32x4_t sum0, sum1, sum2, sum3;

	(void) lag;
	FLAC__ASSERT(lag <= 16);
	FLAC__ASSERT(lag <= data_len);

	sum0 = vdupq_n_f32(0);
	sum1 = vdupq_n_f32(0);
	sum2 = vdupq_n_f32(0);
	sum3 = vdupq_n_f32(0);

	if(limit >= 3) {
		float32x4_t d0, d1, d2, d3, d4;
		float32x4_t sum10, sum11, sum12, sum13, sum20, sum21, sum22, sum23, sum30, sum31, sum32, sum33;
		sum10 = sum11 = sum12 = sum13 = vdupq_n_f32(0);
		sum20 = sum21 = sum22 = sum23 = vdupq_n_f32(0);
		sum30 = sum31 = sum32 = sum33 = vdupq_n_f32(0); 
		d0 = vld1q_f32(data);
		d1 = vld1q_f32(data+4);
		d2 = vld1q_f32(data+8);
		d3 = vld1q_f32(data+12);
		for(i = 0; i <= (limit-3); i += 4) {
			float32x4_t d0_orig = d0;
			d4 = vld1q_f32(data+i+16);
			sum0 = vfmaq_laneq_f32(sum0, d0, d0_orig, 0);
			sum1 = vfmaq_laneq_f32(sum1, d1, d0_orig, 0);
			sum2 = vfmaq_laneq_f32(sum2, d2, d0_orig, 0);
			sum3 = vfmaq_laneq_f32(sum3, d3, d0_orig, 0);
			d0 = vcopyq_laneq_f32(d0_orig, 0, d4, 0);
			sum10 = vfmaq_laneq_f32(sum10, d0, d0_orig, 1);
			sum11 = vfmaq_laneq_f32(sum11, d1, d0_orig, 1);
			sum12 = vfmaq_laneq_f32(sum12, d2, d0_orig, 1);
			sum13 = vfmaq_laneq_f32(sum13, d3, d0_orig, 1);
			d0 = vreinterpretq_f32_f64(vcopyq_laneq_f64(vreinterpretq_f64_f32(d0_orig), 0, vreinterpretq_f64_f32(d4), 0));
			sum20 = vfmaq_laneq_f32(sum20, d0, d0_orig, 2);
			sum21 = vfmaq_laneq_f32(sum21, d1, d0_orig, 2);
			sum22 = vfmaq_laneq_f32(sum22, d2, d0_orig, 2);
			sum23 = vfmaq_laneq_f32(sum23, d3, d0_orig, 2);
			d0 = vcopyq_laneq_f32(d4, 3, d0_orig, 3);
			sum30 = vfmaq_laneq_f32(sum30, d0, d0_orig, 3);
			sum31 = vfmaq_laneq_f32(sum31, d1, d0_orig, 3);
			sum32 = vfmaq_laneq_f32(sum32, d2, d0_orig, 3);
			sum33 = vfmaq_laneq_f32(sum33, d3, d0_orig, 3);
			d0 = d1;
			d1 = d2;
			d2 = d3;
			d3 = d4;
		}
		sum0 = vaddq_f32(sum0, vextq_f32(sum10, sum11, 1));
		sum1 = vaddq_f32(sum1, vextq_f32(sum11, sum12, 1));
		sum2 = vaddq_f32(sum2, vextq_f32(sum12, sum13, 1));
		sum3 = vaddq_f32(sum3, vextq_f32(sum13, sum10, 1));
		sum0 = vaddq_f32(sum0, vextq_f32(sum20, sum21, 2));
		sum1 = vaddq_f32(sum1, vextq_f32(sum21, sum22, 2));
		sum2 = vaddq_f32(sum2, vextq_f32(sum22, sum23, 2));
		sum3 = vaddq_f32(sum3, vextq_f32(sum23, sum20, 2));
		sum0 = vaddq_f32(sum0, vextq_f32(sum30, sum31, 3));
		sum1 = vaddq_f32(sum1, vextq_f32(sum31, sum32, 3));
		sum2 = vaddq_f32(sum2, vextq_f32(sum32, sum33, 3));
		sum3 = vaddq_f32(sum3, vextq_f32(sum33, sum30, 3));
	}
	for(; i <= limit; i++) {
		float32x4_t d0, d1, d2, d3;
		d0 = vld1q_f32(data+i);
		d1 = vld1q_f32(data+i+4);
		d2 = vld1q_f32(data+i+8);
		d3 = vld1q_f32(data+i+12);
		sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
		sum1 = vfmaq_laneq_f32(sum1, d1, d0, 0);
		sum2 = vfmaq_laneq_f32(sum2, d2, d0, 0);
		sum3 = vfmaq_laneq_f32(sum3, d3, d0, 0);
	}

	{
		float32x4_t d0 = vdupq_n_f32(0);
		float32x4_t d1 = vdupq_n_f32(0);
		float32x4_t d2 = vdupq_n_f32(0);
		float32x4_t d3 = vdupq_n_f32(0);
		limit++; if(limit < 0) limit = 0;

		for(; i < (int)data_len; i++) {
			float d = data[i];
			d3 = vextq_f32(d2, d3, 3);
			d2 = vextq_f32(d1, d2, 3);
			d1 = vextq_f32(d0, d1, 3);
			d0 = vextq_f32(d0, d0, 3);
			d0 = vsetq_lane_f32(d, d0, 0);
			sum3 = vfmaq_n_f32(sum3, d3, d);
			sum2 = vfmaq_n_f32(sum2, d2, d);
			sum1 = vfmaq_n_f32(sum1, d1, d);
			sum0 = vfmaq_n_f32(sum0, d0, d);
		}
	}

	vst1q_f32(autoc,   sum0);
	vst1q_f32(autoc+4, sum1);
	vst1q_f32(autoc+8, sum2);
	vst1q_f32(autoc+12,sum3);
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_16_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	int i;
	FLAC__int32 sum;
	const int32x4_t cnt = vdupq_n_s32(-lp_quantization);

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					int16x8_t q0_3, q4_7, q8_11;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_7 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+4));
					q8_11 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+8));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_laneq_s16(      vld2_s16((const FLAC__int16 *)(data+i-12)).val[0], q8_11, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-11)).val[0], q8_11, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-10)).val[0], q8_11, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 9)).val[0], q8_11, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 8)).val[0], q4_7, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 7)).val[0], q4_7, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 6)).val[0], q4_7, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 5)).val[0], q4_7, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 11 */
					int16x8_t q0_3, q4_7;
					int16x4_t q8_9;
					FLAC__int16 q10;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_7 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+4));
					q8_9 = vld1_s16((const FLAC__int16 *)(qlp_coeff+8));
					q10 = *((const FLAC__int16 *)(qlp_coeff+10));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16    (      vld2_s16((const FLAC__int16 *)(data+i-11)).val[0], q10);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i-10)).val[0], q8_9, 2);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i- 9)).val[0], q8_9, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 8)).val[0], q4_7, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 7)).val[0], q4_7, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 6)).val[0], q4_7, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 5)).val[0], q4_7, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 10) {
					int16x8_t q0_3, q4_7;
					int16x4_t q8_9;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_7 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+4));
					q8_9 = vld1_s16((const FLAC__int16 *)(qlp_coeff+8));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_lane_s16 (      vld2_s16((const FLAC__int16 *)(data+i-10)).val[0], q8_9, 2);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i- 9)).val[0], q8_9, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 8)).val[0], q4_7, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 7)).val[0], q4_7, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 6)).val[0], q4_7, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 5)).val[0], q4_7, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i- 1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 9 */
					int16x8_t q0_3, q4_7;
					FLAC__int16 q8;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_7 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+4));
					q8 = *((const FLAC__int16 *)(qlp_coeff+8));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16    (      vld2_s16((const FLAC__int16 *)(data+i-9)).val[0], q8);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-8)).val[0], q4_7, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-7)).val[0], q4_7, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-6)).val[0], q4_7, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-5)).val[0], q4_7, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					int16x8_t q0_3, q4_7;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_7 = vld1q_s16((const FLAC__int16 *)(qlp_coeff+4));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_laneq_s16(      vld2_s16((const FLAC__int16 *)(data+i-8)).val[0], q4_7, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-7)).val[0], q4_7, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-6)).val[0], q4_7, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-5)).val[0], q4_7, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 7 */
					int16x8_t q0_3;
					int16x4_t q4_5;
					FLAC__int16 q6;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_5 = vld1_s16((const FLAC__int16 *)(qlp_coeff+4));
					q6 = *((const FLAC__int16 *)(qlp_coeff+6));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16    (      vld2_s16((const FLAC__int16 *)(data+i-7)).val[0], q6);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i-6)).val[0], q4_5, 2);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i-5)).val[0], q4_5, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 6) {
					int16x8_t q0_3;
					int16x4_t q4_5;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4_5 = vld1_s16((const FLAC__int16 *)(qlp_coeff+4));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_lane_s16 (      vld2_s16((const FLAC__int16 *)(data+i-6)).val[0], q4_5, 2);
						summ = vmlal_lane_s16 (summ, vld2_s16((const FLAC__int16 *)(data+i-5)).val[0], q4_5, 0);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 5 */
					int16x8_t q0_3;
					FLAC__int16 q4;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));
					q4 = *((const FLAC__int16 *)(qlp_coeff+4));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16    (      vld2_s16((const FLAC__int16 *)(data+i-5)).val[0], q4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					int16x8_t q0_3;
					q0_3 = vld1q_s16((const FLAC__int16 *)(qlp_coeff));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_laneq_s16(      vld2_s16((const FLAC__int16 *)(data+i-4)).val[0], q0_3, 6);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q0_3, 4);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_3, 2);
						summ = vmlal_laneq_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 3 */
					int16x4_t q0_1;
					FLAC__int16 q2;
					q0_1 = vld1_s16((const FLAC__int16 *)(qlp_coeff));
					q2 = *((const FLAC__int16 *)(qlp_coeff+2));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16   (      vld2_s16((const FLAC__int16 *)(data+i-3)).val[0], q2);
						summ = vmlal_lane_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_1, 2);
						summ = vmlal_lane_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_1, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 2) {
					int16x4_t q0_1;
					q0_1 = vld1_s16((const FLAC__int16 *)(qlp_coeff));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_lane_s16(      vld2_s16((const FLAC__int16 *)(data+i-2)).val[0], q0_1, 2);
						summ = vmlal_lane_s16(summ, vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0_1, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 1 */
					FLAC__int16 q0 = *((const FLAC__int16 *)(qlp_coeff));

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmull_n_s16(vld2_s16((const FLAC__int16 *)(data+i-1)).val[0], q0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		for(; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 12: sum += qlp_coeff[11] * data[i-12]; /* Falls through. */
				case 11: sum += qlp_coeff[10] * data[i-11]; /* Falls through. */
				case 10: sum += qlp_coeff[ 9] * data[i-10]; /* Falls through. */
				case 9:  sum += qlp_coeff[ 8] * data[i- 9]; /* Falls through. */
				case 8:  sum += qlp_coeff[ 7] * data[i- 8]; /* Falls through. */
				case 7:  sum += qlp_coeff[ 6] * data[i- 7]; /* Falls through. */
				case 6:  sum += qlp_coeff[ 5] * data[i- 6]; /* Falls through. */
				case 5:  sum += qlp_coeff[ 4] * data[i- 5]; /* Falls through. */
				case 4:  sum += qlp_coeff[ 3] * data[i- 4]; /* Falls through. */
				case 3:  sum += qlp_coeff[ 2] * data[i- 3]; /* Falls through. */
				case 2:  sum += qlp_coeff[ 1] * data[i- 2]; /* Falls through. */
				case 1:  sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * data[i-13];
				         sum += qlp_coeff[11] * data[i-12];
				         sum += qlp_coeff[10] * data[i-11];
				         sum += qlp_coeff[ 9] * data[i-10];
				         sum += qlp_coeff[ 8] * data[i- 9];
				         sum += qlp_coeff[ 7] * data[i- 8];
				         sum += qlp_coeff[ 6] * data[i- 7];
				         sum += qlp_coeff[ 5] * data[i- 6];
				         sum += qlp_coeff[ 4] * data[i- 5];
				         sum += qlp_coeff[ 3] * data[i- 4];
				         sum += qlp_coeff[ 2] * data[i- 3];
				         sum += qlp_coeff[ 1] * data[i- 2];
				         sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	int i;
	FLAC__int32 sum;
	const int32x4_t cnt = vdupq_n_s32(-lp_quantization);

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
					int32x4_t q0_3, q4_7, q8_11;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_11 = vld1q_s32(qlp_coeff+8);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_laneq_s32(      vld1q_s32(data+i-12), q8_11, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-11), q8_11, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-10), q8_11, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 9), q8_11, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 8), q4_7, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 7), q4_7, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 6), q4_7, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 5), q4_7, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 11 */
					int32x4_t q0_3, q4_7;
					int32x2_t q8_9;
					FLAC__int32 q10;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_9 = vld1_s32(qlp_coeff+8);
					q10 = *(qlp_coeff+10);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32    (      vld1q_s32(data+i-11), q10);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i-10), q8_9, 1);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i- 9), q8_9, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 8), q4_7, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 7), q4_7, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 6), q4_7, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 5), q4_7, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 10) {
					int32x4_t q0_3, q4_7;
					int32x2_t q8_9;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_9 = vld1_s32(qlp_coeff+8);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_lane_s32 (      vld1q_s32(data+i-10), q8_9, 1);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i- 9), q8_9, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 8), q4_7, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 7), q4_7, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 6), q4_7, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 5), q4_7, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 9 */
					int32x4_t q0_3, q4_7;
					FLAC__int32 q8;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8 = *(qlp_coeff+8);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32    (      vld1q_s32(data+i-9), q8);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-8), q4_7, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-7), q4_7, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-6), q4_7, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-5), q4_7, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
					int32x4_t q0_3, q4_7;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_laneq_s32(      vld1q_s32(data+i-8), q4_7, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-7), q4_7, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-6), q4_7, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-5), q4_7, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 7 */
					int32x4_t q0_3;
					int32x2_t q4_5;
					FLAC__int32 q6;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_5 = vld1_s32(qlp_coeff+4);
					q6 = *(qlp_coeff+6);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32    (      vld1q_s32(data+i-7), q6);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i-6), q4_5, 1);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i-5), q4_5, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 6) {
					int32x4_t q0_3;
					int32x2_t q4_5;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_5 = vld1_s32(qlp_coeff+4);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_lane_s32 (      vld1q_s32(data+i-6), q4_5, 1);
						summ = vmlaq_lane_s32 (summ, vld1q_s32(data+i-5), q4_5, 0);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 5 */
					int32x4_t q0_3;
					FLAC__int32 q4;
					q0_3 = vld1q_s32(qlp_coeff);
					q4 = *(qlp_coeff+4);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32    (      vld1q_s32(data+i-5), q4);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
					int32x4_t q0_3;
					q0_3 = vld1q_s32(qlp_coeff);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_laneq_s32(      vld1q_s32(data+i-4), q0_3, 3);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-3), q0_3, 2);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-2), q0_3, 1);
						summ = vmlaq_laneq_s32(summ, vld1q_s32(data+i-1), q0_3, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 3 */
					int32x2_t q0_1;
					FLAC__int32 q2;
					q0_1 = vld1_s32(qlp_coeff);
					q2 = *(qlp_coeff+2);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32   (      vld1q_s32(data+i-3), q2);
						summ = vmlaq_lane_s32(summ, vld1q_s32(data+i-2), q0_1, 1);
						summ = vmlaq_lane_s32(summ, vld1q_s32(data+i-1), q0_1, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
			else {
				if(order == 2) {
					int32x2_t q0_1;
					q0_1 = vld1_s32(qlp_coeff);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_lane_s32(      vld1q_s32(data+i-2), q0_1, 1);
						summ = vmlaq_lane_s32(summ, vld1q_s32(data+i-1), q0_1, 0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
				else { /* order == 1 */
					FLAC__int32 q0 = *(qlp_coeff);

					for(i = 0; i < (int)data_len-3; i+=4) {
						int32x4_t summ;
						summ = vmulq_n_s32(vld1q_s32(data+i-1), q0);
						summ = vshlq_s32(summ, cnt);
						vst1q_s32(residual+i, vsubq_s32(vld1q_s32(data+i), summ));
					}
				}
			}
		}
		for(; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 12: sum += qlp_coeff[11] * data[i-12]; /* Falls through. */
				case 11: sum += qlp_coeff[10] * data[i-11]; /* Falls through. */
				case 10: sum += qlp_coeff[ 9] * data[i-10]; /* Falls through. */
				case 9:  sum += qlp_coeff[ 8] * data[i- 9]; /* Falls through. */
				case 8:  sum += qlp_coeff[ 7] * data[i- 8]; /* Falls through. */
				case 7:  sum += qlp_coeff[ 6] * data[i- 7]; /* Falls through. */
				case 6:  sum += qlp_coeff[ 5] * data[i- 6]; /* Falls through. */
				case 5:  sum += qlp_coeff[ 4] * data[i- 5]; /* Falls through. */
				case 4:  sum += qlp_coeff[ 3] * data[i- 4]; /* Falls through. */
				case 3:  sum += qlp_coeff[ 2] * data[i- 3]; /* Falls through. */
				case 2:  sum += qlp_coeff[ 1] * data[i- 2]; /* Falls through. */
				case 1:  sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * data[i-13];
				         sum += qlp_coeff[11] * data[i-12];
				         sum += qlp_coeff[10] * data[i-11];
				         sum += qlp_coeff[ 9] * data[i-10];
				         sum += qlp_coeff[ 8] * data[i- 9];
				         sum += qlp_coeff[ 7] * data[i- 8];
				         sum += qlp_coeff[ 6] * data[i- 7];
				         sum += qlp_coeff[ 5] * data[i- 6];
				         sum += qlp_coeff[ 4] * data[i- 5];
				         sum += qlp_coeff[ 3] * data[i- 4];
				         sum += qlp_coeff[ 2] * data[i- 3];
				         sum += qlp_coeff[ 1] * data[i- 2];
				         sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_wide_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	int i;
	FLAC__int64 sum;
	const int64x2_t cnt = vdupq_n_s64(-lp_quantization);

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	if(order <= 12) {
		if(order > 8) { /* order == 9, 10, 11, 12 */
			if(order > 10) { /* order == 11, 12 */
				if(order == 12) {
					int32x4_t q0_3, q4_7, q8_11;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_11 = vld1q_s32(qlp_coeff+8);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_laneq_s32(      vld1_s32(data+i-12), q8_11, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-11), q8_11, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-10), q8_11, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 9), q8_11, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 8), q4_7, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 7), q4_7, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 6), q4_7, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 5), q4_7, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 11 */
					int32x4_t q0_3, q4_7;
					int32x2_t q8_9;
					FLAC__int32 q10;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_9 = vld1_s32(qlp_coeff+8);
					q10 = *(qlp_coeff+10);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32    (      vld1_s32(data+i-11), q10);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i-10), q8_9, 1);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i- 9), q8_9, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 8), q4_7, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 7), q4_7, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 6), q4_7, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 5), q4_7, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
			else { /* order == 9, 10 */
				if(order == 10) {
					int32x4_t q0_3, q4_7;
					int32x2_t q8_9;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8_9 = vld1_s32(qlp_coeff+8);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_lane_s32 (      vld1_s32(data+i-10), q8_9, 1);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i- 9), q8_9, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 8), q4_7, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 7), q4_7, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 6), q4_7, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 5), q4_7, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i- 1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 9 */
					int32x4_t q0_3, q4_7;
					FLAC__int32 q8;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);
					q8 = *(qlp_coeff+8);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32    (      vld1_s32(data+i-9), q8);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-8), q4_7, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-7), q4_7, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-6), q4_7, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-5), q4_7, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
		}
		else if(order > 4) { /* order == 5, 6, 7, 8 */
			if(order > 6) { /* order == 7, 8 */
				if(order == 8) {
					int32x4_t q0_3, q4_7;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_7 = vld1q_s32(qlp_coeff+4);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_laneq_s32(      vld1_s32(data+i-8), q4_7, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-7), q4_7, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-6), q4_7, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-5), q4_7, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 7 */
					int32x4_t q0_3;
					int32x2_t q4_5;
					FLAC__int32 q6;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_5 = vld1_s32(qlp_coeff+4);
					q6 = *(qlp_coeff+6);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32    (      vld1_s32(data+i-7), q6);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i-6), q4_5, 1);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i-5), q4_5, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
			else { /* order == 5, 6 */
				if(order == 6) {
					int32x4_t q0_3;
					int32x2_t q4_5;
					q0_3 = vld1q_s32(qlp_coeff);
					q4_5 = vld1_s32(qlp_coeff+4);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_lane_s32 (      vld1_s32(data+i-6), q4_5, 1);
						summ = vmlal_lane_s32 (summ, vld1_s32(data+i-5), q4_5, 0);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 5 */
					int32x4_t q0_3;
					FLAC__int32 q4;
					q0_3 = vld1q_s32(qlp_coeff);
					q4 = *(qlp_coeff+4);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32    (      vld1_s32(data+i-5), q4);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
		}
		else { /* order == 1, 2, 3, 4 */
			if(order > 2) { /* order == 3, 4 */
				if(order == 4) {
					int32x4_t q0_3;
					q0_3 = vld1q_s32(qlp_coeff);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_laneq_s32(      vld1_s32(data+i-4), q0_3, 3);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-3), q0_3, 2);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-2), q0_3, 1);
						summ = vmlal_laneq_s32(summ, vld1_s32(data+i-1), q0_3, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 3 */
					int32x2_t q0_1;
					FLAC__int32 q2;
					q0_1 = vld1_s32(qlp_coeff);
					q2 = *(qlp_coeff+2);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32   (      vld1_s32(data+i-3), q2);
						summ = vmlal_lane_s32(summ, vld1_s32(data+i-2), q0_1, 1);
						summ = vmlal_lane_s32(summ, vld1_s32(data+i-1), q0_1, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
			else { /* order == 1, 2 */
				if(order == 2) {
					int32x2_t q0_1;
					q0_1 = vld1_s32(qlp_coeff);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_lane_s32(      vld1_s32(data+i-2), q0_1, 1);
						summ = vmlal_lane_s32(summ, vld1_s32(data+i-1), q0_1, 0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
				else { /* order == 1 */
					FLAC__int32 q0 = *(qlp_coeff);

					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t summ;
						summ = vmull_n_s32(vld1_s32(data+i-1), q0);
						summ = vshlq_s64(summ, cnt);
						vst1_s32(residual+i, vsub_s32(vld1_s32(data+i), vmovn_s64(summ)));
					}
				}
			}
		}
		for(; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 12: sum += qlp_coeff[11] * (FLAC__int64)data[i-12]; /* Falls through. */
				case 11: sum += qlp_coeff[10] * (FLAC__int64)data[i-11]; /* Falls through. */
				case 10: sum += qlp_coeff[ 9] * (FLAC__int64)data[i-10]; /* Falls through. */
				case 9:  sum += qlp_coeff[ 8] * (FLAC__int64)data[i- 9]; /* Falls through. */
				case 8:  sum += qlp_coeff[ 7] * (FLAC__int64)data[i- 8]; /* Falls through. */
				case 7:  sum += qlp_coeff[ 6] * (FLAC__int64)data[i- 7]; /* Falls through. */
				case 6:  sum += qlp_coeff[ 5] * (FLAC__int64)data[i- 6]; /* Falls through. */
				case 5:  sum += qlp_coeff[ 4] * (FLAC__int64)data[i- 5]; /* Falls through. */
				case 4:  sum += qlp_coeff[ 3] * (FLAC__int64)data[i- 4]; /* Falls through. */
				case 3:  sum += qlp_coeff[ 2] * (FLAC__int64)data[i- 3]; /* Falls through. */
				case 2:  sum += qlp_coeff[ 1] * (FLAC__int64)data[i- 2]; /* Falls through. */
				case 1:  sum += qlp_coeff[ 0] * (FLAC__int64)data[i- 1];
			}
			residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * (FLAC__int64)data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * (FLAC__int64)data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * (FLAC__int64)data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * (FLAC__int64)data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * (FLAC__int64)data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * (FLAC__int64)data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * (FLAC__int64)data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * (FLAC__int64)data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * (FLAC__int64)data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * (FLAC__int64)data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * (FLAC__int64)data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * (FLAC__int64)data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * (FLAC__int64)data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * (FLAC__int64)data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * (FLAC__int64)data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * (FLAC__int64)data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * (FLAC__int64)data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * (FLAC__int64)data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * (FLAC__int64)data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * (FLAC__int64)data[i-13];
				         sum += qlp_coeff[11] * (FLAC__int64)data[i-12];
				         sum += qlp_coeff[10] * (FLAC__int64)data[i-11];
				         sum += qlp_coeff[ 9] * (FLAC__int64)data[i-10];
				         sum += qlp_coeff[ 8] * (FLAC__int64)data[i- 9];
				         sum += qlp_coeff[ 7] * (FLAC__int64)data[i- 8];
				         sum += qlp_coeff[ 6] * (FLAC__int64)data[i- 7];
				         sum += qlp_coeff[ 5] * (FLAC__int64)data[i- 6];
				         sum += qlp_coeff[ 4] * (FLAC__int64)data[i- 5];
				         sum += qlp_coeff[ 3] * (FLAC__int64)data[i- 4];
				         sum += qlp_coeff[ 2] * (FLAC__int64)data[i- 3];
				         sum += qlp_coeff[ 1] * (FLAC__int64)data[i- 2];
				         sum += qlp_coeff[ 0] * (FLAC__int64)data[i- 1];
			}
			residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
		}
	}
}
#endif /* FLAC__CPU_ARM && FLAC__HAS_NEONINTRIN */
#endif /* FLAC__NO_ASM */
#endif /* FLAC__INTEGER_ONLY_LIBRARY */