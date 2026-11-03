use crate::components::{Composer, Header, ImagesView, Messages};
use crate::images::Studio;
use crate::socket;
use crate::state::Chat;
use leptos::prelude::*;

#[component]
pub fn App() -> impl IntoView {
    let chat = Chat::new();
    let link = socket::connect(chat);
    provide_context(chat);
    provide_context(link);
    provide_context(Studio::new());
    let images = RwSignal::new(false);

    view! {
        <div class="app">
            <nav class="tabs">
                <button class:active=move || !images.get() on:click=move |_| images.set(false)>"💬 Chat"</button>
                <button class:active=move || images.get() on:click=move |_| images.set(true)>"🎨 Images"</button>
            </nav>
            <Show
                when=move || images.get()
                fallback=|| view! { <Header/> <Messages/> <Composer/> }
            >
                <ImagesView/>
            </Show>
        </div>
    }
}
