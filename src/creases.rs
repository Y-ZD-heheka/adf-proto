use crate::{Real, add, dot, len, meshfield, scale, sub};

pub(crate) type Segment = ([[Real; 3]; 2], [[Real; 3]; 2]);

/// Original two-face crease segments and their local search index.
pub struct ReferenceCreases {
    pub(crate) radius: Real,
    cell_width: Real,
    origin: [Real; 3],
    cells: std::collections::HashMap<[i64; 3], Vec<usize>>,
    segments: Vec<Segment>,
}

impl ReferenceCreases {
    /// Index the exact, consistently paired source edges whose dihedral exceeds
    /// `angle` (degrees). `radius` is the largest physical search distance.
    ///
    /// # Panics
    /// Panics for a non-positive or non-finite radius.
    pub fn new(reference: &meshfield::MeshField, radius: Real, angle: Real) -> Self {
        assert!(
            radius.is_finite() && radius > 0.,
            "crease search radius must be finite and positive"
        );
        let extent = sub(reference.aabb_max, reference.aabb_min)
            .into_iter()
            .fold(0., Real::max);
        // A tiny physical query radius should not force a segment into billions of buckets.
        // The exact point-to-segment test still uses the caller's radius.
        let cell_width = radius.max(extent / 1024.);
        let segments = reference.sharp_segments(angle);
        let mut grid = Self {
            radius,
            cell_width,
            origin: reference.aabb_min,
            cells: Default::default(),
            segments,
        };
        for (id, &(edge, _)) in grid.segments.iter().enumerate() {
            let delta = sub(edge[1], edge[0]);
            // Each small bounding box covers one short stretch expanded by the query radius.
            let n = (len(delta) / cell_width).ceil().max(1.0) as usize;
            let mut a = edge[0];
            for i in 1..=n {
                let b = add(edge[0], scale(delta, i as Real / n as Real));
                let lo = grid.cell(std::array::from_fn(|k| a[k].min(b[k]) - radius));
                let hi = grid.cell(std::array::from_fn(|k| a[k].max(b[k]) + radius));
                for x in lo[0]..=hi[0] {
                    for y in lo[1]..=hi[1] {
                        for z in lo[2]..=hi[2] {
                            let entries = grid.cells.entry([x, y, z]).or_default();
                            if entries.last() != Some(&id) {
                                entries.push(id);
                            }
                        }
                    }
                }
                a = b;
            }
        }
        grid
    }

    fn cell(&self, p: [Real; 3]) -> [i64; 3] {
        sub(p, self.origin).map(|v| (v / self.cell_width).floor() as i64)
    }

    pub(crate) fn contains_nearby_segment(&self, p: [Real; 3]) -> bool {
        self.cells.contains_key(&self.cell(p))
    }

    pub(crate) fn target(
        &self,
        p: [Real; 3],
        face_normals: &[[Real; 3]],
        radius: Real,
    ) -> Option<[Real; 3]> {
        let entries = self.cells.get(&self.cell(p))?;
        let mut best = radius * radius;
        let mut result = None;
        for &id in entries {
            let ([a, b], [n, m]) = self.segments[id];
            let ab = sub(b, a);
            let t = (dot(sub(p, a), ab) / dot(ab, ab)).clamp(0., 1.);
            let q = add(a, scale(ab, t));
            let delta = sub(q, p);
            let d2 = dot(delta, delta);
            if d2 >= best {
                continue;
            }
            // Require independent support on both sides of the crease. A nearby
            // smooth sheet (or an unrelated thin parallel wall) supplies only one.
            let mut side = [false; 2];
            for &f in face_normals {
                let (x, y) = (dot(f, n), dot(f, m));
                side[0] |= x > 0.72 && x > y + 0.1;
                side[1] |= y > 0.72 && y > x + 0.1;
            }
            if side == [true, true] {
                best = d2;
                result = Some(q);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wedge() -> Vec<[[Real; 3]; 3]> {
        vec![
            [[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]],
            [[0., 0., 0.], [0., 0., 1.], [2., 0., 0.]],
        ]
    }

    #[test]
    fn a_nearby_smooth_sheet_and_a_nonmanifold_edge_are_not_crease_targets() {
        let source = meshfield::MeshField::new(wedge());
        let grid = ReferenceCreases::new(&source, 0.25, 35.);
        let p = [0.4, 0.05, 0.04];
        let normals = [[0., 0., 1.], [0., 1., 0.]];
        let target = grid.target(p, &normals, 0.25).expect("two supporting facets");
        assert!(len(sub(target, [0.4, 0., 0.])) < 1e-12);
        assert!(grid.target(p, &normals[..1], 0.25).is_none());
        let mut tris = wedge();
        tris.push(tris[0]);
        let source = meshfield::MeshField::new(tris);
        assert!(
            ReferenceCreases::new(&source, 0.25, 35.)
                .target(p, &normals, 0.25)
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "crease search radius")]
    fn zero_search_radius_fails_before_building_buckets() {
        ReferenceCreases::new(&meshfield::MeshField::new(wedge()), 0., 35.);
    }
}
