use crate::Buffer;
use anyhow::{anyhow, Context as _, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLResourceOptions,
};
use std::cell::RefCell;
use std::collections::HashMap;

pub(crate) type Device = ProtocolObject<dyn MTLDevice>;
pub(crate) type State = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const MAX_BUFFER: u64 = 1 << 32;

pub struct Context {
    pub(crate) device: Retained<Device>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub subgroup_size: u32,
    pub max_allocation_size: u64,
    states: RefCell<HashMap<(&'static str, Vec<u32>), State>>,
}

impl Context {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("creating a Metal command queue"))?;
        let max_allocation_size = device.maxBufferLength() as u64;
        Ok(Self {
            device,
            queue,
            subgroup_size: 32,
            max_allocation_size,
            states: RefCell::new(HashMap::new()),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn max_buffer(&self) -> u64 {
        self.max_allocation_size.min(MAX_BUFFER)
    }

    pub fn alloc_host(&self, size: u64) -> Result<Buffer> {
        let options = MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        let buffer = self
            .device
            .newBufferWithLength_options(size.max(16) as usize, options)
            .ok_or_else(|| anyhow!("allocating {size} bytes of shared memory"))?;
        Ok(Buffer::new(buffer, size))
    }

    pub(crate) fn state(&self, name: &'static str, spec: &[u32]) -> Result<State> {
        let key = (name, spec.to_vec());
        if let Some(s) = self.states.borrow().get(&key) {
            return Ok(s.clone());
        }
        let body = skyla_shaders_msl(name)?;
        let mut source = String::from("#define LANES 32\n");
        for (i, v) in spec.iter().enumerate() {
            source.push_str(&format!("#define SPEC_{i} {v}\n"));
        }
        source.push_str(body);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| anyhow!("compiling {name}: {}", e.localizedDescription()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| anyhow!("{name}.metal has no kernel named {name}"))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("pipeline for {name}: {}", e.localizedDescription()))
            .with_context(|| format!("specialization {spec:?}"))?;
        self.states.borrow_mut().insert(key, state.clone());
        Ok(state)
    }
}

fn skyla_shaders_msl(name: &str) -> Result<&'static str> {
    crate::kernels::source(name).ok_or_else(|| anyhow!("no Metal kernel for {name}"))
}
