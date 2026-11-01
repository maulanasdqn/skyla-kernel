use crate::DType;
use anyhow::{bail, Result};
use half::f16;

pub fn dequantize(dtype: DType, src: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        DType::F32 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        DType::F16 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = f16::from_le_bytes(*b).to_f32();
            }
        }
        DType::Q8_0 => {
            for (o, b) in out.chunks_mut(32).zip(src.as_chunks::<34>().0) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for (x, &q) in o.iter_mut().zip(&b[2..]) {
                    *x = d * (q as i8) as f32;
                }
            }
        }
        DType::Q4K => {
            for (o, b) in out.chunks_mut(256).zip(src.as_chunks::<144>().0) {
                dequantize_q4k(b, o);
            }
        }
        DType::Q6K => {
            for (o, b) in out.chunks_mut(256).zip(src.as_chunks::<210>().0) {
                dequantize_q6k(b, o);
            }
        }
        DType::Other(v) => bail!("cannot dequantize ggml type {v}"),
    }
    Ok(())
}

fn q4k_scale_min(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

fn dequantize_q4k(b: &[u8; 144], out: &mut [f32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
    let scales = &b[4..16];
    let qs = &b[16..144];
    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (sc0, m0) = q4k_scale_min(chunk * 2, scales);
        let (sc1, m1) = q4k_scale_min(chunk * 2 + 1, scales);
        let (d0, min0) = (d * sc0 as f32, dmin * m0 as f32);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let o = &mut out[chunk * 64..chunk * 64 + 64];
        for l in 0..32 {
            o[l] = d0 * (q[l] & 0xf) as f32 - min0;
            o[l + 32] = d1 * (q[l] >> 4) as f32 - min1;
        }
    }
}

fn dequantize_q6k(b: &[u8; 210], out: &mut [f32]) {
    let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
    for half in 0..2 {
        let ql = &b[half * 64..half * 64 + 64];
        let qh = &b[128 + half * 32..128 + half * 32 + 32];
        let sc = &b[192 + half * 8..192 + half * 8 + 8];
        let o = &mut out[half * 128..half * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}
