use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// A simple bump allocator that manages a 1MB memory region
/// for allocating custom data structures. This is not a global allocator
/// and only affects code that explicitly uses it.
pub struct BumpAllocator {
    memory: NonNull<u8>,      // Pointer to the memory region
    capacity: usize,          // Total capacity (1MB)
    offset: Cell<usize>,      // Current allocation position
    _marker: PhantomData<u8>, // Mark as !Send and !Sync
}

impl BumpAllocator {
    /// Create a new bump allocator with 1MB of memory
    pub fn new() -> Self {
        let capacity = 1024 * 1024; // 1MB
        let layout = Layout::from_size_align(capacity, 64)
            .expect("Failed to create memory layout");
        
        // Allocate memory using the system allocator
        let memory = unsafe {
            NonNull::new(alloc(layout))
                .expect("Failed to allocate memory for bump allocator")
        };
        
        BumpAllocator {
            memory,
            capacity,
            offset: Cell::new(0),
            _marker: PhantomData,
        }
    }
    
    /// Allocate memory with the given layout
    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align();
        let size = layout.size();
        
        // Calculate aligned offset
        let offset = self.offset.get();
        let aligned_offset = (offset + align - 1) & !(align - 1);
        
        // Check if we have enough space
        if aligned_offset + size > self.capacity {
            return None;
        }
        
        // Update the offset
        self.offset.set(aligned_offset + size);
        
        // Return the allocated memory
        unsafe {
            Some(NonNull::new_unchecked(self.memory.as_ptr().add(aligned_offset)))
        }
    }
    
    /// Reset the allocator, effectively freeing all allocations at once
    pub fn reset(&self) {
        self.offset.set(0);
    }
    
    /// Return current usage in bytes
    pub fn used_bytes(&self) -> usize {
        self.offset.get()
    }
    
    /// Return available bytes
    pub fn available_bytes(&self) -> usize {
        self.capacity - self.offset.get()
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        // Deallocate the entire memory region
        unsafe {
            let layout = Layout::from_size_align(self.capacity, 64)
                .expect("Invalid layout on drop");
            dealloc(self.memory.as_ptr(), layout);
        }
    }
}

/// A wrapper for a typed allocation from the bump allocator
pub struct Bump<T> {
    ptr: NonNull<T>,
    allocator: *const BumpAllocator, // Reference to the allocator that created this
    _marker: PhantomData<T>,
}

impl<T> Bump<T> {
    /// Get a reference to the contained value
    pub fn get(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
    
    /// Get a mutable reference to the contained value
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

/// A simple Vec-like container that uses our bump allocator
pub struct BumpVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    allocator: *const BumpAllocator,
    _marker: PhantomData<T>,
}

impl<T> BumpVec<T> {
    /// Create a new empty vector with the given capacity
    pub fn with_capacity(capacity: usize, allocator: &BumpAllocator) -> Option<Self> {
        if capacity == 0 {
            return Some(BumpVec {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                allocator,
                _marker: PhantomData,
            });
        }
        
        let layout = Layout::array::<T>(capacity).ok()?;
        let ptr = allocator.alloc(layout)?;
        
        Some(BumpVec {
            ptr: ptr.cast(),
            len: 0,
            capacity,
            allocator,
            _marker: PhantomData,
        })
    }
    
    /// Push a value to the end of the vector
    pub fn push(&mut self, value: T) -> bool {
        if self.len == self.capacity {
            return false;
        }
        
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        
        self.len += 1;
        true
    }
    
    /// Get a reference to the element at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            Some(&*self.ptr.as_ptr().add(index))
        }
    }
    
    /// Get a mutable reference to the element at the given index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            Some(&mut *self.ptr.as_ptr().add(index))
        }
    }
    
    /// Return the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Return the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// Example usage
fn main() {
    // Create a bump allocator
    let allocator = BumpAllocator::new();
    
    // Allocate a single value
    let layout = Layout::new::<u64>();
    let ptr = allocator.alloc(layout).expect("Failed to allocate");
    
    // Write a value to it
    unsafe {
        ptr.as_ptr().write(42u8);
        println!("Allocated value: {}", *ptr.as_ptr());
    }
    
    // Create a BumpVec using our allocator
    let mut vec = BumpVec::<i32>::with_capacity(10, &allocator)
        .expect("Failed to create BumpVec");
    
    // Push some values
    for i in 0..10 {
        vec.push(i * 10);
    }
    
    // Read values back
    for i in 0..10 {
        println!("vec[{}] = {}", i, vec.get(i).unwrap());
    }
    
    // Check how much memory we've used
    println!("Used memory: {} bytes", allocator.used_bytes());
    println!("Available memory: {} bytes", allocator.available_bytes());
    
    // Reset the allocator (frees all memory at once)
    allocator.reset();
    println!("After reset - Used memory: {} bytes", allocator.used_bytes());
    
    // Note: After reset, accessing previously allocated objects would be unsafe!
}