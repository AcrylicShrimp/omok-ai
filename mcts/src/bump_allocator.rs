use std::{
    collections::HashSet,
    marker::PhantomData,
    mem::{replace, size_of},
};

pub struct BumpAllocator<T> {
    freed: Vec<*mut T>,
    active_page: BumpAllocatorPage<T>,
    pages: Vec<BumpAllocatorPage<T>>,
    phandom_data: PhantomData<T>,
}

impl<T> BumpAllocator<T> {
    pub const PAGE_SIZE: usize = 1024;

    pub fn new() -> Self {
        Self {
            freed: Vec::new(),
            active_page: BumpAllocatorPage::new(Self::PAGE_SIZE),
            pages: Vec::new(),
            phandom_data: PhantomData,
        }
    }

    pub fn allocate(&mut self, value: T) -> *mut T {
        if let Some(ptr) = self.freed.pop() {
            unsafe { ptr.write(value) };
            return ptr;
        }

        if self.active_page.is_full() {
            self.pages.push(replace(
                &mut self.active_page,
                BumpAllocatorPage::new(Self::PAGE_SIZE),
            ));
        }

        let ptr = self.active_page.allocate();
        unsafe { ptr.write(value) };
        ptr
    }

    pub fn deallocate(&mut self, ptr: *mut T) {
        unsafe { ptr.drop_in_place() };
        self.freed.push(ptr);
    }
}

impl<T> Drop for BumpAllocator<T> {
    fn drop(&mut self) {
        let already_freed = HashSet::from_iter(self.freed.iter().copied());

        self.active_page.drop_all(&already_freed);

        for page in &mut self.pages {
            page.drop_all(&already_freed);
        }
    }
}

pub struct BumpAllocatorPage<T> {
    memory: Vec<u8>,
    capacity: usize,
    offset: usize,
    phandom_data: PhantomData<T>,
}

impl<T> BumpAllocatorPage<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            memory: Vec::with_capacity(size_of::<T>() * capacity),
            capacity,
            offset: 0,
            phandom_data: PhantomData,
        }
    }

    pub fn is_full(&self) -> bool {
        self.offset == self.capacity
    }

    pub fn allocate(&mut self) -> *mut T {
        debug_assert!(self.offset < self.capacity);

        let ptr =
            unsafe { ((&mut self.memory).as_mut_ptr() as *mut T).offset(self.offset as isize) };
        self.offset += 1;
        ptr
    }

    fn drop_all(&mut self, already_freed: &HashSet<*mut T>) {
        let base_ptr = (&mut self.memory).as_mut_ptr() as *mut T;

        for offset in 0..self.offset {
            let ptr = unsafe { base_ptr.offset(offset as isize) };

            if !already_freed.contains(&ptr) {
                unsafe { ptr.drop_in_place() };
            }
        }
    }
}
