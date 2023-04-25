mod bump_allocator;
mod node;

pub use bump_allocator::*;
pub use node::*;

pub struct MCTS<S> {
    root: *mut Node<S>,
    allocator: BumpAllocator<Node<S>>,
}

impl<S> MCTS<S> {
    pub fn new(max_children: usize, root_state: S) -> Self {
        let mut allocator = BumpAllocator::new();
        let root = allocator.allocate(Node::new(None, max_children, root_state));
        Self { root, allocator }
    }

    pub fn root(&self) -> &Node<S> {
        unsafe { &*self.root }
    }

    pub fn transition(&mut self, children_index: usize) {
        let root = unsafe { &mut *self.root };
        let root_children = root.children.read();

        for index in 0..root_children.len() {
            if index == children_index {
                continue;
            }

            let child = unsafe { &mut *(root_children[index] as *mut Node<S>) };
            self.allocator.deallocate(child);
        }

        let new_root = unsafe { &mut *(root_children[children_index] as *mut Node<S>) };
        new_root.parent = None;

        self.allocator.deallocate(self.root);
        self.root = new_root;
    }
}
