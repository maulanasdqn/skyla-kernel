use crate::buffer::Raw;
use crate::context::State;
use crate::{Context, Pipeline};
use anyhow::{anyhow, ensure, Result};
use objc2::rc::autoreleasepool;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDispatchType, MTLSize,
};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr::NonNull;

enum Cmd {
    Dispatch {
        state: State,
        threads: [u32; 3],
        groups: [u32; 3],
        buffers: Vec<Raw>,
        push: Vec<u8>,
    },
    Barrier,
    Mark(&'static str),
}

pub struct Recording {
    cmds: Vec<Cmd>,
    marks: RefCell<Vec<(&'static str, f64)>>,
}

pub struct Recorder<'a> {
    cmds: RefCell<Vec<Cmd>>,
    profile: bool,
    _ctx: PhantomData<&'a Context>,
}

fn size(v: [u32; 3]) -> MTLSize {
    MTLSize {
        width: v[0] as usize,
        height: v[1] as usize,
        depth: v[2] as usize,
    }
}

impl Recorder<'_> {
    pub fn dispatch(&self, p: &Pipeline, groups: [u32; 3], push: &[u32]) {
        self.dispatch_set(p, 0, groups, push);
    }

    pub fn dispatch_set(&self, p: &Pipeline, set: usize, groups: [u32; 3], push: &[u32]) {
        self.cmds.borrow_mut().push(Cmd::Dispatch {
            state: p.state.clone(),
            threads: p.threads,
            groups,
            buffers: p.set(set),
            push: push.iter().flat_map(|w| w.to_le_bytes()).collect(),
        });
    }

    pub fn barrier(&self) {
        self.cmds.borrow_mut().push(Cmd::Barrier);
    }

    pub fn host_barrier(&self) {}

    pub fn mark(&self, label: &'static str) {
        if self.profile {
            self.cmds.borrow_mut().push(Cmd::Mark(label));
        }
    }
}

impl Context {
    pub fn record(&self, profile: bool, f: impl FnOnce(&Recorder)) -> Result<Recording> {
        let recorder = Recorder {
            cmds: RefCell::new(Vec::new()),
            profile,
            _ctx: PhantomData,
        };
        f(&recorder);
        Ok(Recording {
            cmds: recorder.cmds.into_inner(),
            marks: RefCell::new(Vec::new()),
        })
    }

    pub fn replay(&self, r: &Recording) -> Result<()> {
        let segments: Vec<&[Cmd]> = r.cmds.split_inclusive(|c| matches!(c, Cmd::Mark(_))).collect();
        let profiled = segments.len() > 1;
        let mut marks = Vec::new();
        autoreleasepool(|_| -> Result<()> {
            let mut last = None;
            for segment in &segments {
                let cb = self
                    .queue
                    .commandBufferWithUnretainedReferences()
                    .ok_or_else(|| anyhow!("creating a Metal command buffer"))?;
                let enc = cb
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .ok_or_else(|| anyhow!("creating a compute encoder"))?;
                encode(&enc, segment);
                enc.endEncoding();
                cb.commit();
                if profiled {
                    finish(&cb)?;
                    if let Some(Cmd::Mark(label)) = segment.last() {
                        marks.push((*label, (cb.GPUEndTime() - cb.GPUStartTime()) * 1e9));
                    }
                }
                last = Some(cb);
            }
            if let Some(cb) = last {
                finish(&cb)?;
            }
            Ok(())
        })?;
        if profiled {
            *r.marks.borrow_mut() = marks;
        }
        Ok(())
    }

    pub fn marks(&self, r: &Recording) -> Result<Vec<(&'static str, f64)>> {
        let marks = r.marks.borrow().clone();
        ensure!(!marks.is_empty(), "recording was not profiled");
        Ok(marks)
    }
}

fn encode(enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, cmds: &[Cmd]) {
    for cmd in cmds {
        match cmd {
            Cmd::Dispatch {
                state,
                threads,
                groups,
                buffers,
                push,
            } => unsafe {
                enc.setComputePipelineState(state);
                for (i, b) in buffers.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(b), 0, i);
                }
                if !push.is_empty() {
                    let ptr = NonNull::new(push.as_ptr() as *mut _).unwrap();
                    enc.setBytes_length_atIndex(ptr, push.len(), buffers.len());
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(size(*groups), size(*threads));
            },
            Cmd::Barrier => enc.memoryBarrierWithScope(MTLBarrierScope::Buffers),
            Cmd::Mark(_) => {}
        }
    }
}

fn finish(cb: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    cb.waitUntilCompleted();
    match cb.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        _ => Err(anyhow!(
            "Metal command buffer failed: {}",
            cb.error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_default()
        )),
    }
}
