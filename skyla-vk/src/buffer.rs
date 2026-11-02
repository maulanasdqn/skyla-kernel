use crate::Context;
use anyhow::{anyhow, bail, Result};
use ash::vk;
use ash::vk::{BufferUsageFlags, MemoryPropertyFlags};

pub struct Buffer {
    device: ash::Device,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    mapped: Option<*mut u8>,
}

impl Buffer {
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.mapped
            .map(|p| unsafe { std::slice::from_raw_parts_mut(p, self.size as usize) })
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        self.mapped
            .map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, self.size as usize / 4) })
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        self.mapped
            .map(|p| unsafe { std::slice::from_raw_parts_mut(p as *mut f32, self.size as usize / 4) })
    }

    pub fn as_u32_mut(&mut self) -> Option<&mut [u32]> {
        self.mapped
            .map(|p| unsafe { std::slice::from_raw_parts_mut(p as *mut u32, self.size as usize / 4) })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Context {
    pub fn max_buffer(&self) -> u64 {
        self.max_allocation_size
            .min(self.props.limits.max_storage_buffer_range as u64)
    }

    pub fn alloc_host(&self, size: u64) -> Result<Buffer> {
        let excluded = MemoryPropertyFlags::DEVICE_COHERENT_AMD | MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
        let preferred = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
        let memory = self
            .find_memory_type(preferred | MemoryPropertyFlags::HOST_CACHED, excluded)
            .or_else(|| self.find_memory_type(preferred, excluded))
            .ok_or_else(|| anyhow!("no host-visible coherent memory type"))?;
        self.alloc_buffer(size, BufferUsageFlags::STORAGE_BUFFER, memory)
    }

    pub fn find_memory_type(&self, required: MemoryPropertyFlags, excluded: MemoryPropertyFlags) -> Option<u32> {
        self.memory_types()
            .iter()
            .position(|t| t.property_flags.contains(required) && !t.property_flags.intersects(excluded))
            .map(|i| i as u32)
    }

    pub fn alloc_buffer(&self, size: u64, usage: BufferUsageFlags, memory_type: u32) -> Result<Buffer> {
        let device = &self.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        if req.memory_type_bits & (1 << memory_type) == 0 {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("memory type {memory_type} not allowed for this buffer");
        }
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(anyhow!("allocating {size} bytes in memory type {memory_type}: {e}"));
            }
        };
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        let host_visible = self.memory_types()[memory_type as usize]
            .property_flags
            .contains(MemoryPropertyFlags::HOST_VISIBLE);
        let mapped = if host_visible {
            let p = unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }?;
            Some(p as *mut u8)
        } else {
            None
        };
        Ok(Buffer {
            device: device.clone(),
            buffer,
            memory,
            size,
            mapped,
        })
    }
}
