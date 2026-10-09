//! Row-major 2-D grid shared by every module of the native fixed-theatre painters.
//!
//! Row 0 is the first stored row (canvas: north; heightmap stream: south — each user documents its own
//! orientation). Colour images use `Arr2<[u8; 3]>` / `Arr2<[f32; 3]>`; alpha, masks and labels use
//! `Arr2<u8>` / `Arr2<bool>` / `Arr2<i32>`.

#[derive(Clone, Debug, PartialEq)]
pub struct Arr2<T> {
    pub h: usize,
    pub w: usize,
    pub d: Vec<T>,
}

pub type Rgb8 = [u8; 3];
pub type Rgbf = [f32; 3];

impl<T: Copy> Arr2<T> {
    pub fn new(h: usize, w: usize, fill: T) -> Self {
        Arr2 {
            h,
            w,
            d: vec![fill; h * w],
        }
    }

    pub fn from_vec(h: usize, w: usize, d: Vec<T>) -> Self {
        assert_eq!(
            d.len(),
            h * w,
            "Arr2::from_vec: {} elements for {h}x{w}",
            d.len()
        );
        Arr2 { h, w, d }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.d[r * self.w + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: T) {
        let i = r * self.w + c;
        self.d[i] = v;
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[T] {
        &self.d[r * self.w..(r + 1) * self.w]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        let w = self.w;
        &mut self.d[r * w..(r + 1) * w]
    }

    /// Copy of rows `r0..r1`, columns `c0..c1`.
    pub fn crop(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<T> {
        assert!(
            r0 <= r1 && r1 <= self.h && c0 <= c1 && c1 <= self.w,
            "Arr2::crop out of range"
        );
        let mut d = Vec::with_capacity((r1 - r0) * (c1 - c0));
        for r in r0..r1 {
            d.extend_from_slice(&self.d[r * self.w + c0..r * self.w + c1]);
        }
        Arr2 {
            h: r1 - r0,
            w: c1 - c0,
            d,
        }
    }

    /// Overwrite the block whose top-left corner is `(r0, c0)` with `src`.
    pub fn paste(&mut self, r0: usize, c0: usize, src: &Arr2<T>) {
        assert!(
            r0 + src.h <= self.h && c0 + src.w <= self.w,
            "Arr2::paste out of range"
        );
        for r in 0..src.h {
            let dst = (r0 + r) * self.w + c0;
            self.d[dst..dst + src.w].copy_from_slice(src.row(r));
        }
    }

    pub fn map<U: Copy>(&self, mut f: impl FnMut(T) -> U) -> Arr2<U> {
        Arr2 {
            h: self.h,
            w: self.w,
            d: self.d.iter().map(|&v| f(v)).collect(),
        }
    }
}
