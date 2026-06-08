use std::fmt;

#[derive(Debug)]
pub enum PersistenceError {
    EmptyCloud,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidK { k: usize, n: usize },
    EmptyFiltration,
    MatrixSizeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistenceError::EmptyCloud => write!(f, "empty point cloud — at least one point required"),
            PersistenceError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            PersistenceError::InvalidK { k, n } => {
                write!(f, "invalid k for k-NN: k={k} but only {n} points")
            }
            PersistenceError::EmptyFiltration => write!(f, "filtration produced no simplices"),
            PersistenceError::MatrixSizeMismatch { expected, actual } => {
                write!(f, "boundary matrix size mismatch: expected {expected}×{expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PersistenceError {}
