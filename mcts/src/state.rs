pub trait State {
    type Policy: Policy;

    fn is_terminal(&self) -> bool;
    fn policy(&self) -> &Self::Policy;
    fn available_actions_len(&self) -> usize;
    fn is_available_action(&self, action: usize) -> bool;
}

pub trait Policy {
    fn get(&self, action: usize) -> f32;
}
