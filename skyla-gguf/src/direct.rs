use crate::{Gguf, TensorInfo};
use anyhow::Result;
use std::fs::File;
use std::path::Path;

#[cfg(target_vendor = "apple")]
pub fn open(path: &Path) -> Result<File> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    Ok(file)
}

#[cfg(not(target_vendor = "apple"))]
pub fn open(path: &Path) -> Result<File> {
    Ok(File::open(path)?)
}

impl Gguf {
    #[cfg(target_vendor = "apple")]
    pub fn read_into(&self, t: &TensorInfo, out: &mut Vec<u8>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        out.resize(t.n_bytes, 0);
        self.file.read_exact_at(out, t.offset as u64)?;
        Ok(())
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn read_into(&self, t: &TensorInfo, out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        out.extend_from_slice(self.data(t));
        Ok(())
    }
}
