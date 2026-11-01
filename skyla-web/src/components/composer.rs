use crate::socket::Link;
use crate::state::Chat;
use leptos::ev::KeyboardEvent;
use leptos::prelude::*;
use skyla_types::ClientMessage;

#[component]
pub fn Composer() -> impl IntoView {
    let chat = expect_context::<Chat>();
    let link = expect_context::<Link>();
    let draft = RwSignal::new(String::new());

    let send = move || {
        let text = draft.get_untracked();
        let loading = chat.loading.with_untracked(Option::is_some);
        if text.trim().is_empty() || chat.busy.get_untracked() || loading || !chat.connected.get_untracked() {
            return;
        }
        chat.push(true, text.clone());
        chat.push(false, String::new());
        chat.busy.set(true);
        draft.set(String::new());
        link.send(ClientMessage::Chat {
            text,
            settings: chat.settings.get_untracked(),
        });
    };
    let placeholder = move || match chat.connected.get() {
        true => "Message Skyla…  (Enter to send, Shift+Enter for a new line)",
        false => "Disconnected: is `skyla serve` running?",
    };

    view! {
        <footer class="composer">
            <textarea
                rows="1"
                placeholder=placeholder
                prop:value=move || draft.get()
                on:input=move |ev| draft.set(event_target_value(&ev))
                on:keydown=move |ev: KeyboardEvent| {
                    if ev.key() == "Enter" && !ev.shift_key() {
                        ev.prevent_default();
                        send();
                    }
                }
            ></textarea>
            {move || match chat.busy.get() {
                true => view! {
                    <button class="stop" on:click=move |_| link.send(ClientMessage::Stop)>"Stop"</button>
                }
                .into_any(),
                false => view! {
                    <button
                        class="send"
                        disabled=move || draft.with(|d| d.trim().is_empty()) || !chat.connected.get()
                        on:click=move |_| send()
                    >
                        "Send"
                    </button>
                }
                .into_any(),
            }}
        </footer>
    }
}
