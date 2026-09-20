//! Point-cloud generators for closed manifolds.
//!
//! Each generator exposes the shared [`crate::datasets::PointCloudGenerator`]
//! interface and documents its own distribution semantics in its type-level
//! documentation (e.g. [`Sphere`] is uniform with respect to the intrinsic
//! surface measure, while [`KleinBottle`] is uniform in its angle
//! parameters).

mod klein_bottle;
mod sphere;
mod torus;

pub use klein_bottle::KleinBottle;
pub use sphere::Sphere;
pub use torus::Torus;
