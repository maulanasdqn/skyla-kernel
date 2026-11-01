use crate::components::{Composer, Header, Messages};
use crate::socket;
use crate::state::Chat;
use leptos::prelude::*;

#[component]
pub fn App() -> impl IntoView {
    let chat = Chat::new();
    let link = socket::connect(chat);
    provide_context(chat);
    provide_context(link);

    view! {
        <div class="app">
            <Header/>
            <Messages/>
            <Composer/>
        </div>
    }
}
