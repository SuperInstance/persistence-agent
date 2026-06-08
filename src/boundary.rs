use crate::error::PersistenceError;
use crate::vietoris_rips::VietorisRipsComplex;

/// Boundary matrix over Z/2Z with column reduction for persistence computation.
#[derive(Debug, Clone)]
pub struct BoundaryMatrix {
    pub matrix: Vec<Vec<u8>>,
    pub n_simplices: usize,
}

impl BoundaryMatrix {
    /// Build the mod-2 boundary matrix from a Vietoris-Rips complex.
    ///
    /// The columns are indexed by simplices (in the same order as `vr.simplices`),
    /// and rows are indexed by the *simplices one dimension lower*.
    ///
    /// We use a dense representation for clarity: `matrix[col][row]` ∈ {0, 1}.
    /// For each column (a k-simplex), the rows with value 1 correspond to the
    /// (k−1)-faces of that simplex.
    pub fn build(vr: &VietorisRipsComplex) -> Result<Self, PersistenceError> {
        let n = vr.n_simplices();
        if n == 0 {
            return Err(PersistenceError::EmptyFiltration);
        }

        // Map each simplex to its index for fast face lookup
        let mut simplex_index = std::collections::HashMap::new();
        for (i, s) in vr.simplices.iter().enumerate() {
            simplex_index.insert(s.clone(), i);
        }

        let mut matrix = vec![vec![0u8; n]; n];

        for (col_idx, simplex) in vr.simplices.iter().enumerate() {
            if simplex.len() <= 1 {
                // Vertices have no boundary
                continue;
            }
            // All (k−1)-faces: remove one vertex at a time
            for skip in 0..simplex.len() {
                let face: Vec<usize> = simplex
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip)
                    .map(|(_, &v)| v)
                    .collect();
                if let Some(&row_idx) = simplex_index.get(&face) {
                    matrix[col_idx][row_idx] ^= 1;
                }
            }
        }

        Ok(Self {
            matrix,
            n_simplices: n,
        })
    }

    /// Reduce the matrix using the standard algorithm.
    /// After reduction, column `j` is either zero or its lowest nonzero row `low(j)`
    /// is unique. Returns a mapping `low -> j` for non-zero reduced columns.
    pub fn reduce(&mut self) -> std::collections::HashMap<usize, usize> {
        let n = self.n_simplices;
        let mut low_to_col: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for j in 0..n {
            while let Some(lj) = self.low(j) {
                if let Some(&k) = low_to_col.get(&lj) {
                    self.add_columns(k, j);
                } else {
                    low_to_col.insert(lj, j);
                    break;
                }
            }
        }

        low_to_col
    }

    /// Lowest nonzero row in column `j`, or None if the column is zero.
    pub fn low(&self, j: usize) -> Option<usize> {
        self.matrix[j]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &v)| v == 1)
            .map(|(i, _)| i)
    }

    /// Add column `src` to column `dst` (mod 2).
    fn add_columns(&mut self, src: usize, dst: usize) {
        for i in 0..self.n_simplices {
            self.matrix[dst][i] ^= self.matrix[src][i];
        }
    }
}
