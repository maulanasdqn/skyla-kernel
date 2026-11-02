use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

pub(crate) type Raw = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Buffer {
    pub(crate) raw: Raw,
    pub size: u64,
    ptr: *mut u8,
}

impl Buffer {
    pub(crate) fn new(raw: Raw, size: u64) -> Self {
        let ptr = raw.contents().as_ptr() as *mut u8;
        Self { raw, size, ptr }
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) })
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        Some(unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.size as usize / 4) })
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.size as usize / 4) })
    }

    pub fn as_u32_mut(&mut self) -> Option<&mut [u32]> {
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u32, self.size as usize / 4) })
    }
}
