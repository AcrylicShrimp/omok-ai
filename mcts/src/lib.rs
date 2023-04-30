mod bump_allocator;
mod node;

pub use bump_allocator::*;
pub use node::*;

use parking_lot::Mutex;

pub struct MCTS<S> {
    root: *mut Node<S>,
    allocator: Mutex<BumpAllocator<Node<S>>>,
}

impl<S> MCTS<S> {
    pub fn new(max_children: usize, root_state: S) -> Self {
        let mut allocator = BumpAllocator::new();
        let root = allocator.allocate(Node::new(None, max_children, root_state));
        Self {
            root,
            allocator: Mutex::new(allocator),
        }
    }

    pub fn root(&self) -> &Node<S> {
        unsafe { &*self.root }
    }

    pub fn select_leaf(&self, selector: impl Fn(&[*const Node<S>]) -> usize) -> &Node<S> {
        let root = unsafe { &*self.root };
        root.select_leaf(selector)
    }

    pub fn expand<'p, 'c>(&self, node: &'p Node<S>, max_children: usize, state: S) -> &'c Node<S> {
        node.expand(&mut self.allocator.lock(), max_children, state)
    }

    pub fn transition(&mut self, children_index: usize) {
        let allocator = &mut self.allocator.lock();

        let new_root = {
        let root = unsafe { &mut *self.root };
        let root_children = root.children.read();

        for index in 0..root_children.len() {
            if index == children_index {
                continue;
            }

                dealloc_node(root_children[index].ptr as *mut Node<S>, allocator);
        }

            let new_root = unsafe { &mut *(root_children[children_index].ptr as *mut Node<S>) };
        new_root.parent = None;
            new_root as *mut Node<S>
        };

        allocator.deallocate(self.root);
        self.root = new_root;
    }
}

fn dealloc_node<S>(ptr: *mut Node<S>, allocator: &mut BumpAllocator<Node<S>>)
where
    S: State,
{
    {
        let node = unsafe { &mut *ptr };
        let children = node.children.read();
        for child in children.iter() {
            dealloc_node(child.ptr as *mut Node<S>, allocator);
        }
    }
    allocator.deallocate(ptr);
}

unsafe impl<S> Send for MCTS<S> where S: State {}
unsafe impl<S> Sync for MCTS<S> where S: State {}
