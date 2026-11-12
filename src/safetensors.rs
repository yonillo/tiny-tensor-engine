use crate::tensor::Tensor2D;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Metadata for a single tensor inside a `.safetensors` binary file (IC2 Tema 6: Sistemas de Archivos).
#[derive(Clone, Debug)]
pub struct SafeTensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

/// Zero-dependency `.safetensors` binary container reader implemented from scratch in pure Rust.
/// Binary Layout per HuggingFace specification:
/// - Bytes `0..8`: Little-endian `u64` storing byte length `N` of the UTF-8 JSON header.
/// - Bytes `8..8+N`: UTF-8 JSON object mapping tensor names to `{"dtype", "shape", "data_offsets"}`.
/// - Bytes `8+N..EOF`: Raw contiguous IEEE-754 tensor buffers (`BF16`, `F16`, or `F32`).
pub struct SafeTensorsFile {
    pub entries: Vec<SafeTensorEntry>,
    raw_payload: Vec<u8>,
}

impl SafeTensorsFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open safetensors file: {}", e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read safetensors file: {}", e))?;

        if buf.len() < 8 {
            return Err("File too short to be a valid .safetensors container".to_string());
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;

        if buf.len() < 8 + header_len {
            return Err(format!(
                "Truncated .safetensors header: expected {} bytes, file has {}",
                header_len,
                buf.len()
            ));
        }

        let header_str = std::str::from_utf8(&buf[8..8 + header_len])
            .map_err(|e| format!("Invalid UTF-8 in .safetensors header: {}", e))?;

        let entries = Self::parse_header_json(header_str)?;
        let raw_payload = buf[8 + header_len..].to_vec();

        Ok(Self {
            entries,
            raw_payload,
        })
    }

    /// Minimal zero-dependency parser for the flat `.safetensors` JSON header.
    fn parse_header_json(json: &str) -> Result<Vec<SafeTensorEntry>, String> {
        let mut entries = Vec::new();
        let bytes = json.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            // Find key string
            if bytes[i] == b'"' {
                let key_start = i + 1;
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                let key = &json[key_start..i];
                i += 1; // skip closing quote

                // Skip whitespace and colon
                while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
                    i += 1;
                }

                if i < bytes.len() && bytes[i] == b'{' {
                    // Find matching closing brace for this entry object
                    let obj_start = i;
                    let mut depth = 0usize;
                    while i < bytes.len() {
                        if bytes[i] == b'{' {
                            depth += 1;
                        } else if bytes[i] == b'}' {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        i += 1;
                    }
                    let obj_str = &json[obj_start..i];
                    if key != "__metadata__" {
                        if let Some(entry) = Self::parse_entry_object(key, obj_str) {
                            entries.push(entry);
                        }
                    }
                }
            } else {
                i += 1;
            }
        }

        Ok(entries)
    }

    fn parse_entry_object(name: &str, obj: &str) -> Option<SafeTensorEntry> {
        let dtype = Self::extract_string_field(obj, "\"dtype\"")?;
        let shape = Self::extract_usize_array(obj, "\"shape\"")?;
        let offsets = Self::extract_usize_array(obj, "\"data_offsets\"")?;
        if offsets.len() != 2 {
            return None;
        }
        Some(SafeTensorEntry {
            name: name.to_string(),
            dtype,
            shape,
            data_start: offsets[0],
            data_end: offsets[1],
        })
    }

    fn extract_string_field(obj: &str, field: &str) -> Option<String> {
        let pos = obj.find(field)?;
        let rest = &obj[pos + field.len()..];
        let colon = rest.find(':')?;
        let after_colon = &rest[colon + 1..];
        let first_quote = after_colon.find('"')?;
        let after_first = &after_colon[first_quote + 1..];
        let second_quote = after_first.find('"')?;
        Some(after_first[..second_quote].to_string())
    }

    fn extract_usize_array(obj: &str, field: &str) -> Option<Vec<usize>> {
        let pos = obj.find(field)?;
        let rest = &obj[pos + field.len()..];
        let open_br = rest.find('[')?;
        let after_open = &rest[open_br + 1..];
        let close_br = after_open.find(']')?;
        let inner = &after_open[..close_br];
        let mut out = Vec::new();
        for part in inner.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.parse::<usize>().ok()?);
            }
        }
        Some(out)
    }

    /// Loads a 2D tensor by name, converting `BF16` or `F32` little-endian bytes into `Tensor2D`.
    pub fn load_tensor_2d(&self, name: &str) -> Result<Tensor2D, String> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| format!("Tensor '{}' not found in .safetensors", name))?;

        if entry.shape.len() != 2 {
            return Err(format!(
                "Expected 2D tensor for '{}', got shape {:?}",
                name, entry.shape
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let slice = &self.raw_payload[entry.data_start..entry.data_end];

        let mut data = Vec::with_capacity(rows * cols);
        match entry.dtype.as_str() {
            "BF16" => {
                if slice.len() != rows * cols * 2 {
                    return Err("BF16 byte length mismatch".to_string());
                }
                for chunk in slice.chunks_exact(2) {
                    let bits16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let val = f32::from_bits((bits16 as u32) << 16);
                    data.push(val);
                }
            }
            "F32" => {
                if slice.len() != rows * cols * 4 {
                    return Err("F32 byte length mismatch".to_string());
                }
                for chunk in slice.chunks_exact(4) {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    data.push(val);
                }
            }
            other => {
                return Err(format!("Unsupported dtype '{}' in .safetensors", other));
            }
        }

        Ok(Tensor2D::from_vec(rows, cols, data))
    }
}
