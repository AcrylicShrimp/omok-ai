pub fn rotate_90<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in 0..size {
        for j in (0..size).rev() {
            dst[i * size + j] = src[j * size + i];
        }
    }
}

pub fn rotate_180<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in (0..size).rev() {
        for j in (0..size).rev() {
            dst[i * size + j] = src[j * size + i];
        }
    }
}

pub fn rotate_270<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in (0..size).rev() {
        for j in 0..size {
            dst[i * size + j] = src[j * size + i];
        }
    }
}

pub fn flip_horizontal<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in 0..size {
        for j in (0..size).rev() {
            dst[i * size + j] = src[i * size + (size - j - 1)];
        }
    }
}

pub fn flip_vertical<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in (0..size).rev() {
        for j in 0..size {
            dst[i * size + j] = src[(size - i - 1) * size + j];
        }
    }
}
