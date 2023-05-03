pub trait State {
    type PolicyRef<'s>: PolicyRef<'s>
    where
        Self: 's;

    fn is_terminal(&self) -> bool;
    fn policy<'s>(&'s self) -> Self::PolicyRef<'s>;
    fn available_actions_len(&self) -> usize;
    fn is_available_action(&self, action: usize) -> bool;
}

pub trait PolicyRef<'s> {
    fn get(&self, action: usize) -> f32;
}
