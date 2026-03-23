//! Batched command buffer for multi-stage GPU pipelines.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;

/// Batches multiple kernel dispatches into a single Metal command buffer.
pub struct Pipeline {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    retained: Vec<Texture>,
    committed: bool,
}

impl Pipeline {
    /// Creates a new command buffer from `ctx`.
    #[must_use = "pipeline must be committed to execute GPU work"]
    pub fn begin(ctx: &Context) -> Result<Self> {
        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("Failed to create command buffer".into()))?;
        Ok(Self {
            cmd_buf,
            retained: Vec::new(),
            committed: false,
        })
    }

    /// Returns the underlying command buffer.
    pub fn cmd_buf(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.cmd_buf
    }

    /// Retains `tex` until the pipeline completes.
    pub fn retain(&mut self, tex: Texture) {
        self.retained.push(tex);
    }

    /// Retains multiple textures until the pipeline completes.
    pub fn retain_all(&mut self, textures: impl IntoIterator<Item = Texture>) {
        self.retained.extend(textures);
    }

    /// Commits and blocks until GPU completion. Returns retained textures.
    pub fn commit_and_wait(mut self) -> Vec<Texture> {
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();
        self.committed = true;
        std::mem::take(&mut self.retained)
    }

    /// Commits the command buffer without blocking. Call [`Self::wait`] later.
    pub fn commit(&mut self) {
        if !self.committed {
            self.cmd_buf.commit();
            self.committed = true;
        }
    }

    /// Blocks until a previously committed pipeline finishes.
    pub fn wait(&self) {
        if self.committed {
            self.cmd_buf.waitUntilCompleted();
        }
    }

    /// Takes the retained textures out of this pipeline.
    pub fn take_retained(&mut self) -> Vec<Texture> {
        std::mem::take(&mut self.retained)
    }

    /// Returns `true` if [`commit`](Self::commit) has been called.
    pub fn is_committed(&self) -> bool {
        self.committed
    }
}
