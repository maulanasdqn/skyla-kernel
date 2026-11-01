#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecision {
    F16,
    Q8,
    F16KeysQ8Values,
}

impl KvPrecision {
    pub fn keys_q8(self) -> bool {
        self == Self::Q8
    }

    pub fn values_q8(self) -> bool {
        self != Self::F16
    }
}

pub struct KvLayout {
    pub precision: KvPrecision,
    n_layer: u64,
    per_layer: u64,
}

fn width(q8: bool) -> u64 {
    if q8 {
        1
    } else {
        2
    }
}

impl KvLayout {
    pub fn new(precision: KvPrecision, n_layer: usize, max_ctx: usize, kv_dim: usize) -> Self {
        Self {
            precision,
            n_layer: n_layer as u64,
            per_layer: (max_ctx * kv_dim) as u64,
        }
    }

    fn keys_bytes(&self) -> u64 {
        self.n_layer * self.per_layer * width(self.precision.keys_q8())
    }

    fn values_bytes(&self) -> u64 {
        self.n_layer * self.per_layer * width(self.precision.values_q8())
    }

    fn scales_per_layer(&self) -> u64 {
        self.per_layer / 32
    }

    pub fn bytes(&self) -> u64 {
        let scaled = self.precision.keys_q8() as u64 + self.precision.values_q8() as u64;
        self.keys_bytes() + self.values_bytes() + scaled * self.n_layer * self.scales_per_layer() * 2
    }

    pub fn keys(&self, layer: usize) -> u32 {
        (layer as u64 * self.per_layer) as u32
    }

    pub fn values(&self, layer: usize) -> u32 {
        let base = self.keys_bytes() / width(self.precision.values_q8());
        (base + layer as u64 * self.per_layer) as u32
    }

    fn scales_base(&self) -> u64 {
        (self.keys_bytes() + self.values_bytes()) / 2
    }

    pub fn key_scales(&self, layer: usize) -> u32 {
        (self.scales_base() + layer as u64 * self.scales_per_layer()) as u32
    }

    pub fn value_scales(&self, layer: usize) -> u32 {
        let skip = self.precision.keys_q8() as u64 * self.n_layer * self.scales_per_layer();
        (self.scales_base() + skip + layer as u64 * self.scales_per_layer()) as u32
    }
}
