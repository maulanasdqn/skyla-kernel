use super::Graph;
use crate::arena::BATCH;
use crate::kernels::MATMUL_TILE_COLS;
use crate::weights::{LayerWeights, QMatrix};
use skyla_gguf::DType;

impl Graph<'_> {
    fn experts(&self, m: QMatrix, x: (u32, u32), y: (u32, u32)) {
        let units = match m.dtype {
            DType::Q4K => m.n_in / 32,
            DType::Q6K => m.n_in / 64,
            _ => 64,
        };
        let (pipeline, rows_per_group) = self.k.matvec(m.dtype, false).experts(units);
        let groups = m.n_out.div_ceil(rows_per_group);
        let (o, ids) = (m.off, self.a.moe.ids);
        let mut push = match m.dtype {
            DType::Q6K => vec![o[0], o[1], o[2], o[3]],
            _ => vec![o[0], o[1]],
        };
        push.extend([x.0, y.0, m.n_in, m.n_out, ids, x.1, y.1]);
        let slots = self.c.n_expert_used as u32;
        self.r.dispatch_set(pipeline, m.buf, [groups, slots, 1], &push);
    }

    fn grouped(&self, m: QMatrix, x: (u32, u32), x_div: u32, y: (u32, u32)) {
        let mv = self.k.matvec(m.dtype, true);
        let groups = m.n_out.div_ceil(mv.rows_per_group);
        let (o, moe) = (m.off, self.a.moe);
        let push = [
            o[0], o[1], o[2], o[3], x.0, y.0, m.n_in, m.n_out, x.1, y.1, moe.table, moe.list, x_div,
        ];
        let n_expert = self.c.n_expert as u32;
        self.r.dispatch_set(
            mv.grouped(),
            m.buf,
            [groups, n_expert, BATCH as u32 / MATMUL_TILE_COLS],
            &push,
        );
    }

    pub(super) fn moe(&self, lw: &LayerWeights, router: u32) {
        let (a, c, m, b) = (self.a, self.c, self.a.moe, self.b);
        let used = c.n_expert_used as u32;
        let n_embd = c.n_embd as u32;
        let n_expert = c.n_expert as u32;
        let push = [a.n, router, m.logits, n_embd, a.s_embd];
        self.r.dispatch(&self.k.logits, [n_expert, b, 1], &push);
        self.stage("moe logits");
        let push = [m.logits, m.ids, m.weights, n_expert, used];
        self.r.dispatch(&self.k.route, [1, b, 1], &push);
        self.stage("moe route");
        if self.batched {
            let push = [m.ids, m.table, m.list, n_expert, used];
            self.r.dispatch(&self.k.sort, [1, 1, 1], &push);
            self.stage("moe sort");
            self.grouped(lw.gate, (a.n, a.s_embd), used, (m.gate, m.stride));
            self.grouped(lw.up, (a.n, a.s_embd), used, (m.up, m.stride));
        } else {
            self.experts(lw.gate, (a.n, 0), (m.gate, m.stride));
            self.experts(lw.up, (a.n, 0), (m.up, m.stride));
        }
        self.stage("experts gate+up");
        let n = b * used * m.stride;
        self.r
            .dispatch(&self.k.silu, [n.div_ceil(256), 1, 1], &[m.gate, m.up, n, 0]);
        self.stage("silu");
        match self.batched {
            true => self.grouped(lw.down, (m.gate, m.stride), 1, (m.down, a.s_embd)),
            false => self.experts(lw.down, (m.gate, m.stride), (m.down, a.s_embd)),
        }
        self.stage("experts down");
        let push = [a.x, m.down, m.weights, n_embd, a.s_embd, used];
        self.r.dispatch(&self.k.combine, [n_embd.div_ceil(256), b, 1], &push);
        self.stage("moe combine");
    }
}
