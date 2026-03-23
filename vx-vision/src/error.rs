//! Typed error handling for vx-vision.

use std::fmt;

/// All errors that vx-vision operations can produce.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// No Metal-capable GPU found on this system.
    DeviceNotFound,
    /// A named shader function was not found in the compiled metallib.
    ShaderMissing(String),
    /// Metal failed to compile a compute pipeline from a shader function.
    PipelineCompile(String),
    /// GPU buffer allocation failed.
    BufferAlloc { bytes: usize },
    /// Texture dimensions do not match where they must be equal.
    TextureSizeMismatch,
    /// A configuration parameter is out of range or invalid.
    InvalidConfig(String),
    /// GPU runtime error (command buffer, encoder, or execution failure).
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DeviceNotFound => write!(f, "no Metal device found"),
            Error::ShaderMissing(name) => write!(f, "missing shader function '{name}'"),
            Error::PipelineCompile(msg) => write!(f, "pipeline compile: {msg}"),
            Error::BufferAlloc { bytes } => write!(f, "buffer allocation failed ({bytes} bytes)"),
            Error::TextureSizeMismatch => write!(f, "texture size mismatch"),
            Error::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Error::Gpu(msg) => write!(f, "GPU error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Gpu(s)
    }
}

/// Convenience alias used throughout vx-vision.
pub type Result<T> = std::result::Result<T, Error>;
