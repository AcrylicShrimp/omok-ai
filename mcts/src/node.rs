use crate::{bump_allocator::BumpAllocator, state::State, Policy};
use atomic_float::AtomicF32;
use parking_lot::RwLock;
use std::{
    ops::Deref,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

#[derive(Debug)]
pub struct Node<S>
where
    S: State,
{
    pub parent: Option<NodePtr<S>>,
    pub action: Option<usize>,
    pub children: RwLock<Vec<NodePtr<S>>>,
    pub p: f32,
    pub w: AtomicF32,
    pub n: AtomicU64,
    pub v_loss: AtomicU32,
    pub state: S,
}

impl<S> Node<S>
where
    S: State,
{
    pub fn new(parent: Option<NodePtr<S>>, action: Option<usize>, p: f32, state: S) -> Self {
        Self {
            parent,
            action,
            children: RwLock::new(Vec::with_capacity(32)),
            p,
            w: AtomicF32::new(0.0),
            n: AtomicU64::new(0),
            v_loss: AtomicU32::new(0),
            state,
        }
    }

    pub fn select_leaf(&self, selector: impl Fn(&Self, &[NodePtr<S>]) -> usize) -> &Self {
        let mut node = self;
        let mut children = self.children.read();

        loop {
            // If we have not reached the max number of children, return this node, since it is a leaf.
            if children.len() != node.state.available_actions_len() {
                return node;
            }

            if children.len() == 0 {
                return node;
            }

            node.v_loss.fetch_add(1, Ordering::Relaxed);
            let index = selector(node, &children);
            node.v_loss.fetch_sub(1, Ordering::Relaxed);

            node = unsafe { &*children[index].ptr };
            let child_children = node.children.read();
            children = child_children;
        }
    }

    pub fn expand<'p, 'c>(
        &'p self,
        action: usize,
        state: S,
        allocator: &mut BumpAllocator<Self>,
    ) -> Option<&'c Self> {
        let mut children = self.children.write();

        if children.iter().any(|child| child.action == Some(action)) {
            return None;
        }

        let child = allocator.allocate(Self::new(
            Some(NodePtr::new(self)),
            Some(action),
            self.state.policy().get(action),
            state,
        ));

        children.push(NodePtr::new(child));
        Some(unsafe { &*child })
    }

    pub fn propagate(&self, mut w: f32) {
        let mut node = self;

        loop {
            // Update n first; it encourages other threads to select other nodes.
            node.n.fetch_add(1, Ordering::Relaxed);
            node.w.fetch_add(w, Ordering::Relaxed);

            w = -w;

            if let Some(parent) = &node.parent {
                node = &*parent;
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodePtr<S>
where
    S: State,
{
    pub ptr: *const Node<S>,
}

impl<S> NodePtr<S>
where
    S: State,
{
    pub fn new(ptr: *const Node<S>) -> Self {
        Self { ptr }
    }
}

impl<S> Deref for NodePtr<S>
where
    S: State,
{
    type Target = Node<S>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

unsafe impl<S> Send for Node<S> where S: State {}
unsafe impl<S> Send for NodePtr<S> where S: State {}
