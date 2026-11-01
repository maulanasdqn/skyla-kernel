use super::message::MessageView;
use crate::state::Chat;
use leptos::html;
use leptos::prelude::*;

#[component]
pub fn Messages() -> impl IntoView {
    let chat = expect_context::<Chat>();
    let list = NodeRef::<html::Main>::new();

    Effect::new(move |_| {
        chat.tick.track();
        chat.turns.track();
        if let Some(el) = list.get() {
            el.set_scroll_top(el.scroll_height());
        }
    });

    view! {
        <main class="messages" node_ref=list>
            <Show when=move || chat.turns.with(|t| t.is_empty())>
                <div class="empty">
                    <h1>"Skyla"</h1>
                    <p>"DeepSeek-R1 running locally on your Vega iGPU. Ask anything."</p>
                </div>
            </Show>
            <For
                each=move || chat.turns.get()
                key=|turn| turn.id
                children=move |turn| view! { <MessageView turn=turn/> }
            />
        </main>
    }
}
