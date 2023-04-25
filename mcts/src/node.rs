use crate::bump_allocator::BumpAllocator;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Node<S> {
    pub parent: Option<*const Node<S>>,
    pub children: RwLock<Vec<*const Node<S>>>,
    pub max_children: usize,
    pub wins: AtomicU64,
    pub loses: AtomicU64,
    pub visits: AtomicU64,
    pub state: S,
}

impl<S> Node<S> {
    pub fn new(parent: Option<*const Self>, max_children: usize, state: S) -> Self {
        Self {
            parent,
            children: RwLock::new(Vec::with_capacity(32)),
            max_children,
            wins: AtomicU64::new(0),
            loses: AtomicU64::new(0),
            visits: AtomicU64::new(0),
            state,
        }
    }

    pub fn select_leaf(&self, selector: impl Fn(&[*const Self]) -> usize) -> &Self {
        let mut children = self.children.read();

        loop {
            // If we have not reached the max number of children, return this node, since it is a leaf.
            if children.len() != self.max_children {
                return self;
            }

            let index = selector(&children);
            let child = unsafe { &*children[index] };
            let child_children = child.children.read();
            children = child_children;
        }
    }

    pub fn expand(
        &self,
        allocator: &mut BumpAllocator<Self>,
        max_children: usize,
        state: S,
    ) -> *const Self {
        let child = allocator.allocate(Self::new(Some(self), max_children, state));
        let mut children = self.children.write();
        children.push(child);
        child
    }

    pub fn propagate(&self, wins: u64, loses: u64) {
        let mut node = self;

        loop {
            node.wins.fetch_add(wins, Ordering::Relaxed);
            node.loses.fetch_add(loses, Ordering::Relaxed);
            node.visits.fetch_add(1, Ordering::Relaxed);

            if let Some(parent) = node.parent {
                node = unsafe { &*parent };
            } else {
                break;
            }
        }
    }
}
