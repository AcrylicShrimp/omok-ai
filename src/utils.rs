pub fn rotate_90<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in 0..size {
        for j in 0..size {
            dst[i * size + j] = src[(size - j - 1) * size + i];
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
            dst[i * size + j] = src[(size - i - 1) * size + (size - j - 1)];
        }
    }
}

pub fn rotate_270<T>(src: impl AsRef<[T]>, mut dst: impl AsMut<[T]>, size: usize)
where
    T: Copy,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for i in 0..size {
        for j in 0..size {
            dst[i * size + j] = src[j * size + (size - i - 1)];
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rotate_90() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        rotate_90(&src, &mut dst, 2);
        assert_eq!(dst, [3, 1, 4, 2]);
    }

    #[test]
    fn test_rotate_180() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        rotate_180(&src, &mut dst, 2);
        assert_eq!(dst, [4, 3, 2, 1]);
    }

    #[test]
    fn test_rotate_270() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        rotate_270(&src, &mut dst, 2);
        assert_eq!(dst, [2, 4, 1, 3]);
    }

    #[test]
    fn test_flip_horizontal() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        flip_horizontal(&src, &mut dst, 2);
        assert_eq!(dst, [2, 1, 4, 3]);
    }

    #[test]
    fn test_flip_vertical() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        flip_vertical(&src, &mut dst, 2);
        assert_eq!(dst, [3, 4, 1, 2]);
    }
}
